def get_factors(N):
    """
    get all factors of N
    """
    assert isinstance(N, int), f"N={N} must be an integer"
    assert N > 0, f"N={N} must be positive"

    factors = []
    for i in range(1, N + 1):
        if N % i == 0:
            factors.append(i)
    return factors

def get_prime_factors(N):
    """
    get all prime factors of N
    """
    assert isinstance(N, int), f"N={N} must be an integer"
    assert N > 1, f"N={N} must be greater than 1"

    prime_factors = []
    # Check for number of 2s that divide N
    while N % 2 == 0:
        prime_factors.append(2)
        N = N // 2

    # N must be odd at this point, so a skip of 2 (i.e., i = i + 2) can be used
    for i in range(3, int(N**0.5) + 1, 2):
        # While i divides N, append i and divide N
        while N % i == 0:
            prime_factors.append(i)
            N = N // i

    # This condition is to check if N is a prime number greater than 2
    if N > 2:
        prime_factors.append(N)

    return prime_factors

def factorize(N, T, depth=1, path=None, results=None):
    """
    factorize N into T factors
    """
    if path is None:
        path = []

    if results is None:
        results = []

    if T == 1:
        results.append(path + [N])
        return
    for i in get_factors(N):
        factorize(N // i, T - 1, i + 1, path + [i], results)
    return results
