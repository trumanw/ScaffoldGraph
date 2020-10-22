"""
scaffoldgraph.network
"""
from multiprocessing import Pool
from itertools import chain

from loguru import logger
from rdkit.Chem import MolFromSmiles

from .core import MurckoRingFragmenter, MurckoRingSystemFragmenter
from .core import ScaffoldGraph
from .core import Scaffold
from .core import batch_get_scaffold_fragments

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
        i = 0
        for i in range(max_graph_layers_tries):
            if 0 == len(waiting_queue):
                # terminate the loop if no more 
                # scaffolds/molecules are waited to be fragmented
                break
            
            potential_scaffolds_pairs = []
            if len(waiting_queue) > cores:
                batch_size = int(len(waiting_queue) / cores)
                batch_num = cores if len(waiting_queue) % cores == 0 else (cores + 1)

                pool = Pool(cores)
                result_objs = []

                for batch_index in range(batch_num):
                    result_objs.append(
                        pool.apply_async(
                            batch_get_scaffold_fragments, 
                            (waiting_queue[batch_index * batch_size:(batch_index+1) * batch_size], )
                        )
                    )
                batch_potential_scaffolds_pairs = [result.get() for result in result_objs]
                potential_scaffolds_pairs = list(chain(*batch_potential_scaffolds_pairs))
            else:   # no need to be multiprocessing if waiting_queue molecules are less than cores.
                potential_scaffolds_pairs = batch_get_scaffold_fragments(waiting_queue)
                
            waiting_queue_in_smiles = []
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
                            waiting_queue_in_smiles.append(parent_smi)

            # transfer smiles to rdmol and add to the waiting_queue
            waiting_queue = [MolFromSmiles(smi) for smi in waiting_queue_in_smiles]
        
        if i == max_graph_layers_tries:
            logger.warning(f'Loop of iteratively building graph has exceeded the maximum limitation {i}')