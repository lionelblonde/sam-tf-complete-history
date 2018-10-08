from imitation.common.console_util import columnize


def test_columnize():
    tuples = [(1, 2, 3),
              (4, 5, 6)]
    print(columnize(names=['col1', 'col2', 'col3'],
                    tuples=tuples,
                    widths=[10, 20, 5]))


if __name__ == "__main__":
    test_columnize()
