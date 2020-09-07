"""
scaffoldgraph.prioritization.generic_rules

Generic rules for defining custom rulesets* for prioritizing
scaffolds during scaffold tree construction.

*scaffoldgraph.prioritization.prioritization_ruleset.ScaffoldRuleSet

Rule Prefix definitions:
------------------------
SCP - Scaffold property (parent scaffold)
RRP - Removed ring property
RSP - Property of ring system of removed ring before removal
"""

from rdkit.Chem import MolFromSmarts

from itertools import chain, compress
from abc import abstractmethod

from scaffoldgraph.core.fragment import collect_linker_atoms

from .prioritization_rules import BaseScaffoldFilterRule


__all__ = [
    'SCPNumLinkerBonds',
    'SCPDelta',
    'SCPAbsDelta',
    'SCPNumAromaticRings',
    'SCPNumHetAtoms',
    'SCPNumNAtoms',
    'SCPNumOAtoms',
    'SCPNumSAtoms',
    'SCPNumXAtoms',
    'RRPRingSize',
    'RRPLinkerLength',
    'RRPHetAtomLinked',
    'RRPLinkerLengthX',
    'RRPNumHetAtoms',
    'RRPNumNAtoms',
    'RRPNumOAtoms',
    'RRPNumSAtoms',
    'RRPNumXAtoms',
    'RRPRingSizeX',
    'RSPAbsDelta',
    'RSPDelta',
    'RSPNumAromaticRings',
    'RSPNumHetAtoms',
    'RSPNumNAtoms',
    'RSPNumOAtoms',
    'RSPNumRings',
    'RSPNumSAtoms',
    'RSPNumXAtoms',
    'Tiebreaker'
]


class _MinMaxScaffoldFilterRule(BaseScaffoldFilterRule):
    """Abstract base class for generic rules where 'min' or 'max' filtering can be specified"""

    _f = {'min', 'max'}

    def __init__(self, min_max='min'):
        assert min_max in self._f, f'function must be min or max'
        self.func = min if min_max == 'min' else max

    def filter(self, child, parents):
        props = [self.get_property(child, s) for s in parents]
        val = self.func(props)
        return list(compress(parents, [True if p == val else False for p in props]))

    @abstractmethod
    def get_property(self, child, parent):
        raise NotImplementedError()

    @property
    def name(self):
        return '{}'.format(
            self.__class__.__name__
        )


class SCPNumLinkerBonds(_MinMaxScaffoldFilterRule):
    """Filter by number of linker bonds in the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of acyclic linker bonds

    Specify 'max' to prioritize scaffolds with the largest
    number of acyclic linker bonds

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    acyc_linker_smarts = MolFromSmarts('*!@!=!#*')

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        matches = parent.mol.GetSubstructMatches(self.acyc_linker_smarts)
        return len(matches)


class SCPDelta(_MinMaxScaffoldFilterRule):
    """Filter by the delta value of the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    delta value

    Specify 'max' to prioritize scaffolds with the largest
    delta value

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        nr = parent.rings.count
        rb = list(chain(*parent.rings.bond_rings))
        nrrb = len(rb) - len(set(rb))
        delta = nrrb - (nr - 1)
        return delta


class SCPAbsDelta(SCPDelta):
    """Filter by the absolute delta value of the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    absolute delta value

    Specify 'max' to prioritize scaffolds with the largest
    absolute delta value

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        return abs(super().get_property(child, parent))


class SCPNumHetAtoms(_MinMaxScaffoldFilterRule):
    """Filter by the number of heteroatoms in the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of heteroatoms

    Specify 'max' to prioritize scaffolds with the largest
    number of heteroatoms

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        parent_atomic_nums = [a.GetAtomicNum() for a in parent.atoms]
        num_het = len([a for a in parent_atomic_nums if a != 1 and a != 6])
        return num_het


class SCPNumAromaticRings(_MinMaxScaffoldFilterRule):
    """Filter by the number of aromatic rings in the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of aromatic rings

    Specify 'max' to prioritize scaffolds with the largest
    number of aromatic rings

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        aro_r = [x.aromatic for x in parent.rings]
        return aro_r.count(True)


class _SCPAtomicNumRule(_MinMaxScaffoldFilterRule):

    def __init__(self, min_max, atomic_num):
        super().__init__(min_max)
        self.atomic_num = atomic_num

    def get_property(self, child, parent):
        parent_atomic_nums = [a.GetAtomicNum() for a in parent.atoms]
        return parent_atomic_nums.count(self.atomic_num)


class SCPNumOAtoms(_SCPAtomicNumRule):
    """Filter by the number of oxygen in the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of oxygen atoms

    Specify 'max' to prioritize scaffolds with the largest
    number of oxygen atoms

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 8)


class SCPNumNAtoms(_SCPAtomicNumRule):
    """Filter by the number of nitrogen atoms in the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of nitrogen atoms

    Specify 'max' to prioritize scaffolds with the largest
    number of nitrogen atoms

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 7)


class SCPNumSAtoms(_SCPAtomicNumRule):
    """Filter by the number of sulphur atoms in the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of sulphur atoms

    Specify 'max' to prioritize scaffolds with the largest
    number of sulphur atoms

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 16)


class SCPNumXAtoms(_SCPAtomicNumRule):
    """Filter by the number atoms with atomic number X in
    the parent scaffold

    Specify 'min' to prioritize scaffolds with the smallest
    number of X atoms

    Specify 'max' to prioritize scaffolds with the largest
    number of X atoms

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    atomic_num : (int) atomic number for prioritization
    """

    def __init__(self, min_max, atomic_num):
        super().__init__(min_max, atomic_num)


class RRPRingSize(_MinMaxScaffoldFilterRule):
    """Filter by the size of the removed ring

    Specify 'min' to prioritize scaffolds where the smallest
    ring has been removed

    Specify 'max' to prioritize scaffolds where the largest
    ring has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        return removed_ring.size


class RRPNumHetAtoms(_MinMaxScaffoldFilterRule):
    """Filter by the number of heteroatoms in the removed ring

    Specify 'min' to prioritize scaffolds where the ring with
    the least heteroatoms has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most heteroatoms has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        ring_atomic_nums = [a.GetAtomicNum() for a in removed_ring.atoms]
        return len([a for a in ring_atomic_nums if a != 1 and a != 6])


class _RRPAtomicNumRule(_MinMaxScaffoldFilterRule):

    def __init__(self, min_max, atomic_num):
        super().__init__(min_max)
        self.atomic_num = atomic_num

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        ring_atomic_nums = [a.GetAtomicNum() for a in removed_ring.atoms]
        return ring_atomic_nums.count(self.atomic_num)


class RRPNumOAtoms(_RRPAtomicNumRule):
    """Filter by the number of oxygen atoms in the removed ring

    Specify 'min' to prioritize scaffolds where the ring with
    the least oxygen atoms has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most oxygen atoms has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 8)


class RRPNumNAtoms(_RRPAtomicNumRule):
    """Filter by the number of nitrogen atoms in the removed ring

    Specify 'min' to prioritize scaffolds where the ring with
    the least nitrogen atoms has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most nitrogen atoms has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 7)


class RRPNumSAtoms(_RRPAtomicNumRule):
    """Filter by the number of sulphur atoms in the removed ring

    Specify 'min' to prioritize scaffolds where the ring with
    the least sulphur atoms has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most sulphur atoms has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 16)


class RRPNumXAtoms(_RRPAtomicNumRule):
    """Filter by the number of atoms with atomic number X in
     the removed ring

    Specify 'min' to prioritize scaffolds where the ring with
    the least X atoms has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most X atoms has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    atomic_num : (int) atomic number for prioritization
    """

    def __init__(self, min_max, atomic_num):
        super().__init__(min_max, atomic_num)


class RRPHetAtomLinked(_MinMaxScaffoldFilterRule):
    """Filter by whether the removed rings linker is attached to
    a ring hetero atom at either end of the linker

    Specify 'min' to prioritize scaffolds where the removed rings
    linker is not attached to a heteroatom

    Specify 'max' to prioritize scaffolds where the removed rings
    linker is attached to a heteroatom

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        linker, ra = set(), set()
        removed_ring = child.rings[parent.removed_ring_idx]
        attachments = removed_ring.get_attachment_points()
        for attachment in attachments:
            ra.update(collect_linker_atoms(
                child.mol.GetAtomWithIdx(attachment), linker, False
            ))
        atomic_nums = [child.atoms[x].GetAtomicNum() for x in ra]
        return len([a for a in atomic_nums if a != 1 and a != 6]) > 0


class RRPRingSizeX(RRPRingSize):
    """Filter by the size X of the removed ring where the
    ring size X is specified

    Specify 'min' to prioritize scaffolds where the removed
    rings size is != to X

    Specify 'max' to prioritize scaffolds where the removed
    rings size == to X

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    size (int) ring size for prioritization
    """

    def __init__(self, min_max, size):
        super().__init__(min_max)
        self.size = size

    def get_property(self, child, parent):
        rs = super().get_property(child, parent)
        return rs == self.size


class RRPLinkerLength(_MinMaxScaffoldFilterRule):
    """Filter by the size of the removed rings linker

    Specify 'min' to prioritize scaffolds where the ring with
    the smallest linker has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the largest linker has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        linker = set()
        removed_ring = child.rings[parent.removed_ring_idx]
        attachments = removed_ring.get_attachment_points()
        for attachment in attachments:
            collect_linker_atoms(
                child.mol.GetAtomWithIdx(attachment), linker, False
            )
        return len(linker)


class RRPLinkerLengthX(RRPLinkerLength):
    """Filter by the size X of the removed rings linker
    where the linker size X is specified

    Specify 'min' to prioritize scaffolds where the removed
    rings linker size is != X

    Specify 'max' to prioritize scaffolds where the removed
    rings linker size == to X

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    length (int) linker size for prioritization
    """

    def __init__(self, min_max, length):
        super().__init__(min_max)
        self.length = length

    def get_property(self, child, parent):
        linker_length = super().get_property(child, parent)
        return linker_length == self.length


class RSPDelta(_MinMaxScaffoldFilterRule):
    """Filter by the delta value of the removed rings
    ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the smallest ring system delta value has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the largest ring system delta value has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        system = removed_ring.get_ring_system()
        nr = system.num_rings
        rb = list(chain(*[x.bix for x in system]))
        nrrb = len(rb) - len(set(rb))
        delta = nrrb - (nr - 1)
        return delta


class RSPAbsDelta(RSPDelta):
    """Filter by the absolute delta value of the removed rings
    ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the smallest ring system abs delta value has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the largest ring system abs delta value has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        return abs(super().get_property(child, parent))


class RSPNumRings(_MinMaxScaffoldFilterRule):
    """Filter by the size of the removed rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the smallest ring system has been removed (num rings)

    Specify 'max' to prioritize scaffolds where the ring with
    the largest ring system has been removed (num rings)

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        system = removed_ring.get_ring_system()
        return system.num_rings


class RSPNumAromaticRings(_MinMaxScaffoldFilterRule):
    """Filter by the number of aromatic rings in the removed
     rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the least aromatic rings in its ring system has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most aromatic rings in its ring system has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        system = removed_ring.get_ring_system()
        aro_rings = [x.aromatic for x in system].count(True)
        return aro_rings


class RSPNumHetAtoms(_MinMaxScaffoldFilterRule):
    """Filter by the number of heteroatoms in the removed
     rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the least heteroatoms in its ring system has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most heteroatoms in its ring system has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max)

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        system = removed_ring.get_ring_system()
        sys_atomic_nums = [a.GetAtomicNum() for a in system.atoms]
        return len([a for a in sys_atomic_nums if a != 1 and a != 6])


class _RSPAtomicNumRule(_MinMaxScaffoldFilterRule):

    def __init__(self, min_max, atomic_num):
        super().__init__(min_max)
        self.atomic_num = atomic_num

    def get_property(self, child, parent):
        removed_ring = child.rings[parent.removed_ring_idx]
        system = removed_ring.get_ring_system()
        sys_atomic_nums = [a.GetAtomicNum() for a in system.atoms]
        return sys_atomic_nums.count(self.atomic_num)


class RSPNumNAtoms(_RSPAtomicNumRule):
    """Filter by the number of nitrogen atoms in the removed
     rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the least nitrogen atoms in its ring system has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most nitrogen atoms in its ring system has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 7)


class RSPNumOAtoms(_RSPAtomicNumRule):
    """Filter by the number of oxygen atoms in the removed
     rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the least oxygen atoms in its ring system has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most oxygen atoms in its ring system has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 8)


class RSPNumSAtoms(_RSPAtomicNumRule):
    """Filter by the number of sulphur atoms in the removed
     rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the least sulphur atoms in its ring system has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most sulphur atoms in its ring system has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max):
        super().__init__(min_max, 16)


class RSPNumXAtoms(_RSPAtomicNumRule):
    """Filter by the number of atoms with the atomic number X
     in the removed rings ring system

    Specify 'min' to prioritize scaffolds where the ring with
    the least X atoms in its ring system has been removed

    Specify 'max' to prioritize scaffolds where the ring with
    the most X atoms in its ring system has been removed

    Parameters
    ----------
    min_max : (str ('min' or 'max')) specify 'min' or 'max'
        to define the function used to prioritize scaffolds
        based on the returned property
    """

    def __init__(self, min_max, atomic_num):
        super().__init__(min_max, atomic_num)


class Tiebreaker(BaseScaffoldFilterRule):
    """Tie-breaker rule (alphabetical)

    In the case where multiple scaffolds are left after all
    rules have been evaluated, sort the scaffolds by their
    canonical SMILES and keep the first.
    """

    def filter(self, child, parents):
        return [sorted(parents, key=lambda p: p.smiles)[0]]

    @property
    def name(self):
        return '{}'.format(
            self.__class__.__name__
        )
