import networkx as nx
import platform
import matplotlib.path as mpltPath




class Simulation:
    """ called once holds important information about the
        simulation
    """

    def __init__(self, name, path, end_time, time_step, pluri_div_thresh, diff_div_thresh, pluri_to_diff, size,
                 diff_surround_value, functions, parallel, max_fgf4, bounds, death_threshold, move_time_step,
                 move_max_time, spring_constant, friction, energy_kept, neighbor_distance):

        """ Initialization function for the simulation setup.
            name: the simulation name
            path: the path to save the simulation information to
            start_time: the start time for the simulation
            end_time: the end time for the simulation
            time_step: the time step to increment the simulation by
            pluri_div_thresh: threshold for pluripotent cells to divide
            diff_div_thresh:  threshold for differentiated cells to divide
            pluri_to_diff: threshold for pluripotent cells to differentiate
            size: the size of the grid (dimension, rows, columns)
            spring_max: represents the max distance for spring and other interactions
            diff_surround_value: the amount of differentiated cells needed to surround
                a pluripotent cell inducing its differentiation
            functions: the boolean functions as a string from Model_Setup
            itrs: the max amount of times optimize will run
            error: the max error allowed for optimize
            parallel: whether some aspects are run on the gpu
            max_fgf4: the limit a patch will hold for fgf4
            bounds: the bounds of the simulation
            spring_constant: strength of spring
        """
        self.name = name
        self.path = path
        self.end_time = end_time
        self.time_step = time_step
        self.pluri_div_thresh = pluri_div_thresh
        self.diff_div_thresh = diff_div_thresh
        self.pluri_to_diff = pluri_to_diff
        self.size = size
        self.diff_surround_value = diff_surround_value
        self.functions = functions
        self.parallel = parallel
        self.max_fgf4 = max_fgf4
        self.bounds = bounds
        self.death_threshold = death_threshold
        self.move_time_step = move_time_step
        self.move_max_time = move_max_time
        self.spring_constant = spring_constant
        self.friction = friction
        self.energy_kept = energy_kept
        self.neighbor_distance = neighbor_distance

        # the array that represents the grid and all its patches
        self.grid = np.zeros(self.size)

        # counts how many times an image is created for making videos
        self.image_counter = 0

        # keeps a running count of the time
        self.time_counter = 0.0

        # array to hold all of the stem cell objects
        self.objects = np.array([])

        # graph representing all the objects and their connections to other objects
        self.network = nx.Graph()

        # holds the objects until they are added or removed from the simulation
        self._objects_to_remove = np.array([])
        self._objects_to_add = np.array([])

        # keeps track of the current cell ID
        self._current_ID = 0

        # which file separator to use
        if platform.system() == "Windows":
            # windows
            self._sep = "\\"
        else:
            # linux/unix
            self._sep = "/"

        self.boundary = mpltPath.Path(self.bounds)