import numpy as np
import copy
from numba import jit, cuda, prange
import math
import cv2

import backend


# This is an archive of methods that aren't currently being used

@backend.record_time
def alt_cell_motility(simulation):
    """ gives the cells a motive force depending on
        set rules for the cell types expect these rules
        are very similar to NetLogo
    """
    # this is the motility force of the cells
    motility_force = 0.000000002

    # loop over all of the cells
    for i in range(simulation.number_cells):
        # see if the cell is moving or not
        if simulation.cell_motion[i]:
            # get the neighbors of the cell if the cell is actively moving
            neighbors = simulation.neighbor_graph.neighbors(i)

            # if cell is surrounded by other cells, inhibit the motion
            if len(neighbors) >= 6:
                simulation.cell_motion[i] = False

            # if not, calculate the active movement for the step
            else:
                if simulation.cell_states[i] == "Differentiated":
                    # if there is a nanog high cell nearby, move away from it
                    if not np.isnan(simulation.cell_nearest_nanog[i]):
                        nearest_index = int(simulation.cell_nearest_nanog[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force * -1

                    # if no nearby nanog high cells, move randomly
                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if the cell is gata6 high and nanog low
                elif simulation.cell_fds[i][2] > simulation.cell_fds[i][3]:
                    # if there is a differentiated cell nearby, move toward it
                    if not np.isnan(simulation.cell_nearest_diff[i]):
                        nearest_index = int(simulation.cell_nearest_diff[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force

                    # if no nearby differentiated cells, move randomly
                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if the cell is nanog high and gata6 low
                elif simulation.cell_fds[i][3] > simulation.cell_fds[i][2]:
                    # if there is a nanog high cell nearby, move toward it
                    if not np.isnan(simulation.cell_nearest_nanog[i]):
                        nearest_index = int(simulation.cell_nearest_nanog[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force * 0.8
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force * 0.2

                    # if there is a gata6 high cell nearby, move away from it
                    elif not np.isnan(simulation.cell_nearest_gata6[i]):
                        nearest_index = int(simulation.cell_nearest_gata6[i])
                        vector = simulation.cell_locations[nearest_index] - simulation.cell_locations[i]
                        normal = backend.normal_vector(vector)
                        simulation.cell_motility_force[i] += normal * motility_force * -1

                    else:
                        simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force

                # if both gata6/nanog high or both low, move randomly
                else:
                    simulation.cell_motility_force[i] += backend.random_vector(simulation) * motility_force


@backend.record_time
def nearest(simulation):
    """ looks at cells within a given radius a determines
        the closest cells of certain types
    """
    # radius of search for nearest cells
    nearest_distance = 0.000015

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(nearest, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        nearest.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, bin_locations, max_cells = backend.assign_bins(simulation, nearest_distance, nearest.max_cells)

    # update the value of the max number of cells in a bin and double it
    nearest.max_cells = max_cells

    # turn the following arrays into True/False
    if_diff = simulation.cell_states == "Differentiated"
    gata6_high = simulation.cell_fds[:, 2] == 1
    nanog_high = simulation.cell_fds[:, 3] == 1

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        locations_cuda = cuda.to_device(simulation.cell_locations)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        distance_cuda = cuda.to_device(nearest_distance)
        if_diff_cuda = cuda.to_device(if_diff)
        gata6_high_cuda = cuda.to_device(gata6_high)
        nanog_high_cuda = cuda.to_device(nanog_high)
        nearest_gata6_cuda = cuda.to_device(simulation.cell_nearest_gata6)
        nearest_nanog_cuda = cuda.to_device(simulation.cell_nearest_nanog)
        nearest_diff_cuda = cuda.to_device(simulation.cell_nearest_diff)

        # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
        tpb = 72
        bpg = math.ceil(simulation.number_cells / tpb)

        # call the cuda kernel with given parameters
        backend.nearest_gpu[bpg, tpb](locations_cuda, bins_cuda, bins_help_cuda, distance_cuda, if_diff_cuda,
                                      gata6_high_cuda, nanog_high_cuda, nearest_gata6_cuda, nearest_nanog_cuda,
                                      nearest_diff_cuda)

        # return the arrays back from the gpu
        gata6 = nearest_gata6_cuda.copy_to_host()
        nanog = nearest_nanog_cuda.copy_to_host()
        diff = nearest_diff_cuda.copy_to_host()

    # call the cpu version
    else:
        gata6, nanog, diff = backend.nearest_cpu(simulation.number_cells, simulation.cell_locations, bins, bins_help,
                                                 nearest_distance, if_diff, gata6_high, nanog_high,
                                                 simulation.cell_nearest_gata6, simulation.cell_nearest_nanog,
                                                 simulation.cell_nearest_diff)

    # revalue the array holding the indices of nearest cells of given type
    simulation.cell_nearest_gata6 = gata6
    simulation.cell_nearest_nanog = nanog
    simulation.cell_nearest_diff = diff


@jit(nopython=True, parallel=True)
def nearest_cpu(number_cells, cell_locations, bins, bins_help, distance, if_diff, gata6_high, nanog_high, nearest_gata6,
                nearest_nanog, nearest_diff):
    """ this is the just-in-time compiled version of nearest
        that runs in parallel on the cpu
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
        # offset bins by 2 to avoid missing cells that fall outside the space
        block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # initialize these variables with essentially nothing values and the distance as an initial comparison
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = np.nan, np.nan, np.nan
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance * 2, distance * 2, distance * 2

        # loop over the bin the cell is in and the surrounding bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through that bin
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if that cell is within the search radius and not the same cell
                        mag = np.linalg.norm(cell_locations[current] - cell_locations[focus])
                        if mag <= distance and focus != current:
                            # update the nearest differentiated cell first
                            if if_diff[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = mag

                            # update the nearest gata6 high cell making sure not nanog high
                            elif gata6_high[current]:
                                if not nanog_high[current]:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_gata6_dist:
                                        nearest_gata6_index = current
                                        nearest_gata6_dist = mag

                            # update the nearest nanog high cell
                            elif nanog_high[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = mag

        # update the nearest cell of desired type
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index

    # return the updated edges
    return nearest_gata6, nearest_nanog, nearest_diff


@cuda.jit
def nearest_gpu(cell_locations, bins, bins_help, distance, if_diff, gata6_high, nanog_high, nearest_gata6,
                nearest_nanog, nearest_diff):
    """ this is the cuda kernel for the nearest function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        # offset bins by 2 to avoid missing cells that fall outside the space
        x = int(cell_locations[focus][0] / distance[0]) + 2
        y = int(cell_locations[focus][1] / distance[0]) + 2
        z = int(cell_locations[focus][2] / distance[0]) + 2

        # initialize these variables with essentially nothing values and the distance as an initial comparison
        nearest_gata6_index, nearest_nanog_index, nearest_diff_index = np.nan, np.nan, np.nan
        nearest_gata6_dist, nearest_nanog_dist, nearest_diff_dist = distance[0] * 2, distance[0] * 2, distance[0] * 2

        # loop over the bin the cell is in and the surrounding bins
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of cells for the current bin
                    bin_count = bins_help[x + i][y + j][z + k]

                    # go through that bin determining if a cell is a neighbor
                    for l in range(bin_count):
                        # get the index of the current cell in question
                        current = bins[x + i][y + j][z + k][l]

                        # check to see if that cell is within the search radius and not the same cell
                        mag = magnitude(cell_locations[focus], cell_locations[current])
                        if mag <= distance[0] and focus != current:
                            # update the nearest differentiated cell first
                            if if_diff[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_diff_dist:
                                    nearest_diff_index = current
                                    nearest_diff_dist = mag

                            # update the nearest gata6 high cell making sure not nanog high
                            elif gata6_high[current]:
                                if not nanog_high[current]:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_gata6_dist:
                                        nearest_gata6_index = current
                                        nearest_gata6_dist = mag

                            # update the nearest nanog high cell
                            elif nanog_high[current]:
                                # if it's closer than the last cell, update the nearest magnitude and index
                                if mag < nearest_nanog_dist:
                                    nearest_nanog_index = current
                                    nearest_nanog_dist = mag

        # update the nearest cell of certain types
        nearest_gata6[focus] = nearest_gata6_index
        nearest_nanog[focus] = nearest_nanog_index
        nearest_diff[focus] = nearest_diff_index


# Find the nearest pluripotent cell within a fixed radius that is not part of the same component of the underlying
# graph of all pluripotent cells. Used to represent the movement of pluripotent clusters. (not in use)
# functions.nearest_cluster(simulation)

@backend.record_time
def nearest_cluster(simulation):
    """ find the nearest nanog high cells outside the cluster
        that the cell is currently in
    """
    # radius of search for the nearest pluripotent cell not in the same cluster
    nearest_distance = 0.0002

    # create a copy of the neighbor graph and get the edges
    nanog_graph = copy.deepcopy(simulation.neighbor_graph)
    edges = np.array(nanog_graph.get_edgelist())

    # create an array to hold edges to delete
    length = len(edges)
    delete = np.zeros(length, dtype=int)
    delete_help = np.zeros(length, dtype=bool)

    # use parallel jit function to find differentiated/gata6 nodes/edges
    if length != 0:
        delete, delete_help = backend.remove_gata6_edges(length, simulation.cell_fds, edges, delete, delete_help)

    # only delete edges meant to be deleted by the help array
    delete = delete[delete_help]
    nanog_graph.delete_edges(delete)

    # get the membership to corresponding clusters
    members = np.array(nanog_graph.clusters().membership)

    # if a static variable has not been created to hold the maximum number of cells in a bin, create one
    if not hasattr(nearest_cluster, "max_cells"):
        # begin with a low number of cells that can be revalued if the max number of cells exceeds this value
        nearest_cluster.max_cells = 5

    # calls the function that generates an array of bins that generalize the cell locations in addition to a
    # creating a helper array that assists the search method in counting cells in a particular bin
    bins, bins_help, bin_locations, max_cells = backend.assign_bins(simulation, nearest_distance,
                                                                    nearest_cluster.max_cells)

    # update the value of the max number of cells in a bin and double it
    nearest_cluster.max_cells = max_cells

    # turn the following array into True/False
    if_nanog = simulation.cell_fds[:, 3] == 1

    # call the nvidia gpu version
    if simulation.parallel:
        # turn the following into arrays that can be interpreted by the gpu
        distance_cuda = cuda.to_device(nearest_distance)
        bins_cuda = cuda.to_device(bins)
        bins_help_cuda = cuda.to_device(bins_help)
        locations_cuda = cuda.to_device(simulation.cell_locations)
        if_nanog_cuda = cuda.to_device(if_nanog)
        cell_nearest_cluster_cuda = cuda.to_device(simulation.cell_nearest_cluster)
        members_cuda = cuda.to_device(members)

        # allocate threads and blocks for gpu memory "threads per block" and "blocks per grid"
        tpb = 72
        bpg = math.ceil(simulation.number_cells / tpb)

        # call the cuda kernel with given parameters
        backend.nearest_cluster_gpu[bpg, tpb](locations_cuda, bins_cuda, bins_help_cuda, distance_cuda, if_nanog_cuda,
                                              cell_nearest_cluster_cuda, members_cuda)

        # return the array back from the gpu
        nearest_cell = cell_nearest_cluster_cuda.copy_to_host()

    # call the cpu version
    else:
        nearest_cell = backend.nearest_cluster_cpu(simulation.number_cells, simulation.cell_locations, bins, bins_help,
                                                   nearest_distance, if_nanog, simulation.cell_nearest_cluster, members)

    # revalue the array holding the indices of nearest nanog high cells outside cluster
    simulation.cell_cluster_nearest = nearest_cell


@jit(nopython=True, parallel=True)
def remove_gata6_edges(length, fds, edges, delete, delete_help):
    """ used by the outside cluster function to
        remove differentiated/gata6 high edges.
    """
    # go through edges
    for i in prange(length):
        # add to the delete array if either node is differentiated/gata6 high
        if fds[edges[i][0]][2] or fds[edges[i][1]][2]:
            delete[i] = i
            delete_help[i] = 1

    return delete, delete_help


@jit(nopython=True, parallel=True)
def nearest_cluster_cpu(number_cells, cell_locations, bins, bins_help, distance, if_nanog, cell_cluster_nearest,
                        members):
    """ This is the Numba optimized version of
        the nearest_cluster function.
    """
    # loops over all cells, with the current cell index being the focus
    for focus in prange(number_cells):
        if if_nanog[focus]:
            # offset bins by 2 to avoid missing cells that fall outside the space
            block_location = cell_locations[focus] // distance + np.array([2, 2, 2])
            x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

            # initialize these variables with essentially nothing values and the distance as an initial comparison
            nearest_cell_index = np.nan
            nearest_cell_dist = distance * 2

            # loop over the bin the cell is in and the surrounding bin
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        # get the count of cells for the current bin
                        bin_count = bins_help[x + i][y + j][z + k]

                        # go through that bin
                        for l in range(bin_count):
                            # get the index of the current cell in question
                            current = bins[x + i][y + j][z + k][l]

                            # make sure not differentiated, not same cell, and not in the same cluster
                            if if_nanog[current] and focus != current and members[focus] != members[current]:
                                # check to see if that cell is within the search radius and not the same cell
                                mag = np.linalg.norm(cell_locations[current] - cell_locations[focus])
                                if mag <= distance:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_cell_dist:
                                        nearest_cell_index = current
                                        nearest_cell_dist = mag

            # update the nearest cell of desired type
            cell_cluster_nearest[focus] = nearest_cell_index

    # return the updated edges
    return cell_cluster_nearest


@cuda.jit
def nearest_cluster_gpu(cell_locations, bins, bins_help, distance, if_nanog, cell_nearest_cluster, members):
    """ this is the cuda kernel for the nearest function
        that runs on a NVIDIA gpu
    """
    # get the index in the array
    focus = cuda.grid(1)

    # checks to see that position is in the array
    if focus < cell_locations.shape[0]:
        if if_nanog[focus]:
            # offset bins by 2 to avoid missing cells that fall outside the space
            x = int(cell_locations[focus][0] / distance[0]) + 2
            y = int(cell_locations[focus][1] / distance[0]) + 2
            z = int(cell_locations[focus][2] / distance[0]) + 2

            # initialize these variables with essentially nothing values and the distance as an initial comparison
            nearest_cell_index = np.nan
            nearest_cell_dist = distance[0] * 2

            # loop over the bin the cell is in and the surrounding bin
            for i in range(-1, 2):
                for j in range(-1, 2):
                    for k in range(-1, 2):
                        # get the count of cells for the current bin
                        bin_count = bins_help[x + i][y + j][z + k]

                        # go through that bin
                        for l in range(bin_count):
                            # get the index of the current cell in question
                            current = bins[x + i][y + j][z + k][l]

                            # make sure not differentiated, not same cell, and not in the same cluster
                            if if_nanog[current] and focus != current and members[focus] != members[current]:
                                # check to see if that cell is within the search radius and not the same cell
                                mag = magnitude(cell_locations[current], cell_locations[focus])
                                if mag <= distance[0]:
                                    # if it's closer than the last cell, update the nearest magnitude and index
                                    if mag < nearest_cell_dist:
                                        nearest_cell_index = current
                                        nearest_cell_dist = mag

            # update the nearest cell of desired type
            cell_nearest_cluster[focus] = nearest_cell_index


@backend.record_time
def alt_highest_fgf4(simulation):
    """ Search for the highest concentrations of
        fgf4 within a fixed radius
    """
    for focus in range(simulation.number_cells):
        # offset bins by 2 to avoid missing points
        block_location = simulation.cell_locations[focus] // simulation.diffuse_radius + np.array([2, 2, 2])
        x, y, z = int(block_location[0]), int(block_location[1]), int(block_location[2])

        # create a holder for nearby diffusion points, a counter for the number, and values
        holder = np.zeros((4, 3))
        count = 0
        values = np.zeros(4)

        # loop over the bins that surround the current bin
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # get the count of points in a bin
                    bin_count = simulation.diffuse_bins_help[x + i][y + j][z + k]

                    # go through the bin determining if a bin is within the search radius
                    for l in range(bin_count):
                        # get the indices of the current point in question
                        x_ = int(simulation.diffuse_bins[x + i][y + j][z + k][l][0])
                        y_ = int(simulation.diffuse_bins[x + i][y + j][z + k][l][1])
                        z_ = int(simulation.diffuse_bins[x + i][y + j][z + k][l][2])

                        # check to see if that point is within the search radius
                        m = np.linalg.norm(simulation.diffuse_locations[x_][y_][z_] - simulation.cell_locations[focus])
                        if m < simulation.diffuse_radius:
                            # if it is, add it to the holder and its value to values
                            holder[count][0] = x_
                            holder[count][1] = y_
                            holder[count][2] = z_
                            values[count] = simulation.fgf4_values[x_][y_][z_]
                            count += 1

        # get the sum of the array
        sum_ = np.sum(values)

        # calculate probability of moving toward each point
        if sum_ == 0:
            # update the highest fgf4 diffusion point
            simulation.cell_highest_fgf4[focus][0] = np.nan
            simulation.cell_highest_fgf4[focus][1] = np.nan
            simulation.cell_highest_fgf4[focus][2] = np.nan
        else:
            probs = values / sum_

            # randomly choose based on a custom distribution the diffusion point to move to
            thing = np.random.choice(np.arange(4), p=probs)

            # get the index
            index = holder[thing]

            # update the highest fgf4 diffusion point
            simulation.cell_highest_fgf4[focus][0] = index[0]
            simulation.cell_highest_fgf4[focus][1] = index[1]
            simulation.cell_highest_fgf4[focus][2] = index[2]



# import numpy as np
# np.savetxt("nearest_point_fgf4_step_48.csv", simulation.fgf4_values[:, :, 0], delimiter=",")
# np.savetxt("distance_dependent_fgf4_step_48.csv", simulation.fgf4_alt[:, :, 0], delimiter=",")
#
# print(simulation.fgf4_alt[:, :, 0])


def alt_step_image(simulation):
    # get the size of the array used for imaging in addition to the scale factor
    pixels = simulation.image_quality
    scale = pixels / simulation.size[0]
    x_size = pixels
    y_size = math.ceil(scale * simulation.size[1])

    # normalize the concentration values and multiple by 255 to create grayscale image
    grad_image = simulation.fgf4_values[:, :, 0] * (255 / simulation.max_concentration)
    grad_image = grad_image.astype(np.uint8)

    # recolor the grayscale image into a colormap and resize to match the cell space array
    grad_image = cv2.applyColorMap(grad_image, cv2.COLORMAP_OCEAN)
    grad_image = cv2.resize(grad_image, (y_size, x_size), interpolation=cv2.INTER_NEAREST)

    # flip and rotate to turn go from (y, x) to (x, y)
    grad_image = cv2.rotate(grad_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    grad_image = cv2.flip(grad_image, 0)

    # normalize the concentration values and multiple by 255 to create grayscale image
    grad_alt_image = simulation.fgf4_alt[:, :, 0] * (255 / simulation.max_concentration)
    grad_alt_image = grad_alt_image.astype(np.uint8)

    # recolor the grayscale image into a colormap and resize to match the cell space array
    grad_alt_image = cv2.applyColorMap(grad_alt_image, cv2.COLORMAP_OCEAN)
    grad_alt_image = cv2.resize(grad_alt_image, (y_size, x_size), interpolation=cv2.INTER_NEAREST)

    # flip and rotate to turn go from (y, x) to (x, y)
    grad_alt_image = cv2.rotate(grad_alt_image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    grad_alt_image = cv2.flip(grad_alt_image, 0)

    # combine the to images side by side if including gradient
    image = np.concatenate((grad_image, grad_alt_image), axis=1)

    # flip the image horizontally so origin is bottom left
    image = cv2.flip(image, 0)

    # save the image as a png
    image_path = simulation.images_path + simulation.name + "_image_" + str(int(simulation.current_step)) + ".png"
    cv2.imwrite(image_path, image)

# Locate the diffusion point (within a fixed radius) that has the highest FGF4 concentration. Can be used to
# approximate the chemotactic movement of cells. (not in use)
# functions.highest_fgf4(simulation)

# Places all of the diffusion points into bins so that the model can use a bin sorting method when determining
# cell motility based on diffusion point locations. (not in use)
# functions.setup_diffusion_bins(simulation)