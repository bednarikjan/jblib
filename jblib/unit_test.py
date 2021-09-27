# Global test index.
TEST_IDX = 1


def next_test(msg):
    """ Prints the name of the test and increments `TEST_IDX` counter.

    Args:
        msg (str): Name of the test.
    """
    global TEST_IDX
    headline = 'TEST {} - {}'.format(TEST_IDX, msg)
    print(headline)
    print(''.join(['='] * len(headline)))
    TEST_IDX += 1
