def sum(n):
    '''
    Calculates the sum in range 1 and n.


    Parameters
    ----------
    n : int
        Positive integer less of equal 10^15 

    Returns
    -------
    int
        Sum of range 1 to n, or 0 if given wrong input

    '''
    if  isinstance(n, int) & (n <= 10000000000000000000000000) & ( n > 0):
        return n*(n+1)/2
    else:
        return(0)


number = 10000000000000000000000000
sum(number)


