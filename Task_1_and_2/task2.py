import numpy as np 


def mark_island_seen(map_coords, coord, M, N):
    '''
    Returnes map with removed island by given coordinates
    (changes all connected 1s to 0s)
    
    Parameters
    ----------
    map_coords : np.array
        2D Array with the elements of 0s and 1s
    coord : tuple
        Tuple with row and column values of the island's element
    M : int
        number of rows in the map
    N : int
        number of columns in the map
        
    Returns
    -------
    map_coords : np.array
        2D array with removed island

    '''
    row, col = coord
    map_coords[row][col] = 0

    lands_nearby = [(row-1, col), (row, col-1), (row+1, col), (row, col+1)]
    for r, c in lands_nearby:
        if (
                r < M and
                c < N and
                r >= 0 and
                c >= 0 and
                map_coords[r,c] == 1
            ):
                map_coords = mark_island_seen(map_coords, (r,c), M, N)
    return map_coords
            

def count_islands(map_coord, M, N):
    '''
    Return the number of islands of 1s in a 2D array

    Parameters
    ----------
    map_coord : np.array
        2D Array with the elements of 0s and 1s.
    M : int
        number of rows in the map
    N : int
        number of columns in the map

    Returns
    -------
    islands_count : int
        Number of islands

    '''
    islands_count = 0
    for coord, x in np.ndenumerate(map_coord):
        if x == 1:
            map_coord = mark_island_seen(map_coord, coord, M, N)
            islands_count += 1
    return islands_count
               


# testing the program:
map_coord1 = np.array([[0,1,0],[0,0,0],[0,1,1]])
assert count_islands(map_coord1, 3, 3) == 2, 'should be 2'

map_coord2 = np.array([[0,0,0,1],[0,0,1,0],[0,1,0,0]])
assert count_islands(map_coord2, 3, 4) == 3, 'should be 3'

map_coord3 = np.array([[0,0,0,1],[0,0,1,1],[0,1,0,1]])
assert count_islands(map_coord3, 3, 4) == 2, 'should be 2'












  

