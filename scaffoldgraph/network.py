"""
scaffoldgraph.network
"""
from multiprocessing import Pool

from loguru import logger
from rdkit.Chem import MolFromSmiles

from .core import MurckoRingFragmenter, MurckoRingSystemFragmenter
from .core import ScaffoldGraph
from .core import Scaffold

class ScaffoldNetwork(ScaffoldGraph):
    """
    Class representing a scaffold network.

    Explore scaffold-space through the iterative removal of available rings,
    generating all possible sub-scaffolds for a set of input molecules.
    The output is a directed acyclic graph of molecular scaffolds.

    Examples
    --------
    Create a ScaffoldNetwork from a SMILES file.

    >>> import scaffoldgraph as sg
    >>> network = sg.ScaffoldNetwork.from_smiles_file('my_file.smi', progress=True)
    >>> network.num_scaffold_nodes
    100

    Create a ScaffoldNetwork from an SDF.

    >>> network = sg.ScaffoldNetwork.from_sdf('my_file.sdf', progress=True)

    If the SDF is zipped:

    >>> network = sg.ScaffoldNetwork.from_sdf('my_file.sdf.gz', zipped=True)

    Get scaffold nodes:

    >>> list(network.get_scaffold_nodes())
    ['O=C(OCOC(=O)c1cccc2ncn(Cc3ccccc3)c12)OC1CCCCC1',
    'O=C(OCOC(=O)c1cccc2nc[nH]c12)OC1CCCCC1',
    ...]

    Include node attributes:

    >>> list(network.get_scaffold_nodes(data=True))
    [('O=C(OCOC(=O)c1cccc2ncn(Cc3ccccc3)c12)OC1CCCCC1', {'type': 'scaffold', 'hierarchy': 4}),
    ('O=C(OCOC(=O)c1cccc2nc[nH]c12)OC1CCCCC1', {'type': 'scaffold', 'hierarchy': 3}),
    ...]

    Get molecule nodes (use data=True to get attributes):

    >>> list(network.get_molecule_nodes())
    ['DB00006',
     'DB00007',
     'DB00014',
     ...]


    References
    ----------
    .. [1] Varin, T., Schuffenhauer, A., Ertl, P., and Renner, S. (2011). Mining for bioactive
           scaffolds with scaffold networks: Improved compound set enrichment from primary screening data.
           Journal of Chemical Information and Modeling, 51(7), 1528–1538.

    See Also
    --------
    ScaffoldGraph
    ScaffoldTree
    HierS

    """
    def __init__(self, graph=None, **kwargs):
        """Initialize a ScaffoldNetwork.

        Parameters
        ----------
        graph : input graph, optional
            Data to initialize graph. If None (default) an empty
            graph is created. The data can be any format that is supported
            by the ``to_networkx_graph()`` function, currently including
            edge list, dict of dicts, dict of lists, NetworkX graph,
            NumPy matrix or 2d ndarray, SciPy sparse matrix,
            or PyGraphviz graph. This argument is passed to the networkx
            DiGraph constructor.

        """
        super(ScaffoldNetwork, self).__init__(graph, MurckoRingFragmenter(), 'network')

    def _recursive_constructor(self, child):
        parents = (p for p in self.fragmenter.fragment(child) if p)
        for parent in parents:
            if parent in self.nodes:
                self.add_scaffold_edge(parent, child)
            else:
                self.add_scaffold_node(parent)
                self.add_scaffold_edge(parent, child)
                if parent.rings.count > 1:
                    self._recursive_constructor(parent)


class HierS(ScaffoldGraph):
    """
    Class representing a HierS type scaffold network.

    Explore scaffold-space through the iterative removal of available rings,
    generating all possible sub-scaffolds without dissecting fused ring-systems.

    Notes
    -----
    A HierS type network differs from a conventional scaffold network, through construction.
    When fragmenting molecules the HierS constructor does not attempt to break fused ring
    systems.

    Examples
    --------
    Create a HierS network from a SMILES file.

    >>> import scaffoldgraph as sg
    >>> network = sg.HierS.from_smiles_file('my_file.smi', progress=True)
    >>> network.num_scaffold_nodes
    92

    Create a HierS netwoek from an SDF.

    >>> network = sg.HierS.from_sdf('my_file.sdf', progress=True)

    If the SDF is zipped:

    >>> network = sg.HierS.from_sdf('my_file.sdf.gz', zipped=True)

    Get scaffold nodes:

    >>> list(network.get_scaffold_nodes())
    ['O=C(OCOC(=O)c1cccc2ncn(Cc3ccccc3)c12)OC1CCCCC1',
    'O=C(OCOC(=O)c1cccc2nc[nH]c12)OC1CCCCC1',
    ...]

    Include node attributes:

    >>> list(network.get_scaffold_nodes(data=True))
    [('O=C(OCOC(=O)c1cccc2ncn(Cc3ccccc3)c12)OC1CCCCC1', {'type': 'scaffold', 'hierarchy': 4}),
    ('O=C(OCOC(=O)c1cccc2nc[nH]c12)OC1CCCCC1', {'type': 'scaffold', 'hierarchy': 3}),
    ...]

    Get molecule nodes (use data=True to get attributes):

    >>> list(network.get_molecule_nodes())
    ['DB00006',
     'DB00007',
     'DB00014',
     ...]

    References
    ----------
    .. [1] Wilkens, S., Janes, J., and Su, A. (2005). HierS: Hierarchical Scaffold Clustering
           Using Topological Chemical Graphs. Journal of Medicinal Chemistry, 48(9), 3182-3193.

    See Also
    --------
    ScaffoldGraph
    ScaffoldNetwork
    ScaffoldTree

    """
    def __init__(self, graph=None, **kwargs):
        """Initialize a HierS network.

        Parameters
        ----------
        graph : input graph, optional
            Data to initialize graph. If None (default) an empty
            graph is created. The data can be any format that is supported
            by the ``to_networkx_graph()`` function, currently including
            edge list, dict of dicts, dict of lists, NetworkX graph,
            NumPy matrix or 2d ndarray, SciPy sparse matrix,
            or PyGraphviz graph. This argument is passed to the networkx
            DiGraph constructor.

        """
        super(HierS, self).__init__(graph, MurckoRingSystemFragmenter(), 'hiers')

    def _recursive_constructor(self, child):
        parents = (p for p in self.fragmenter.fragment(child) if p)
        for parent in parents:
            if parent in self.nodes:
                self.add_scaffold_edge(parent, child)
            else:
                self.add_scaffold_node(parent)
                self.add_scaffold_edge(parent, child)
                if parent.ring_systems.count > 1:
                    self._recursive_constructor(parent)

    def _multiprocess_constructor(self, waiting_queue, cores=4, max_graph_layers_tries=1000):
        for i in range(max_graph_layers_tries):
            if 0 == len(waiting_queue):
                # terminate the loop if no more 
                # scaffolds/molecules are waited to be fragmented
                break

            pool = Pool(cores)
            potential_scaffolds_pairs = pool.map(fragment, waiting_queue)
            
            waiting_queue = []
            for child, parents in potential_scaffolds_pairs:
                child_scaffold = Scaffold(MolFromSmiles(child))
                
                for parent_smi in parents:
                    parent_scaffold = Scaffold(MolFromSmiles(parent_smi))
                    if parent_scaffold in self.nodes:
                        self.add_scaffold_edge(parent_scaffold, child_scaffold)
                    else:
                        self.add_scaffold_node(parent_scaffold)
                        self.add_scaffold_edge(parent_scaffold, child_scaffold)
                        if parent_scaffold.ring_systems.count > 1:
                            waiting_queue.append(parent_scaffold.mol)
        
        if i == max_graph_layers_tries:
            logger.warning(f'Loop of iteratively building graph has exceeded the maximum limitation {i}')

def fragment(scaffold_rdmol):
    import scaffoldgraph as sg
    fragmenter = sg.HierS().fragmenter
    scaffold_obj = Scaffold(scaffold_rdmol)
    fragments = fragmenter.fragment(scaffold_obj)
    return scaffold_obj.get_canonical_identifier(), [f.get_canonical_identifier() for f in fragments]