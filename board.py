import random
import re
from enum import Enum
from tkinter import Tk, Button, W
from typing import Iterable

import numpy as np

from util import MinesweeperHelper, AllGroups, AllClusters, MineGroup

# https://www.askpython.com/python-modules/tkinter/tkinter-colors
COLORS = {
    0: 'black',
    1: 'navy',
    2: 'dodgerblue',
    3: 'lightseagreen',
    4: 'yellowgreen',
    5: 'greenyellow',
    6: 'yellow',
    7: 'gold',
    8: 'orange'
}
NUM2WORD = {
    0: 'zero',
    1: 'one',
    2: 'two',
    3: 'three',
    4: 'four',
    5: 'five',
    6: 'six',
    7: 'seven',
    8: 'eight'
}
LEFT_MOUSE_BTN = '<Button-1>'
MIDDLE_MOUSE_BTN = '<Button-2>'
RIGHT_MOUSE_BTN = '<Button-3>'


class Mode(Enum):
    SET_MINES = 0
    SET_START = 1
    TRY = 2


class Board:
    master: Tk
    size: int
    mines: np.ndarray
    buttons: np.ndarray
    opened: np.ndarray
    flagged: np.ndarray
    start_field: tuple[int, int]
    mode: Mode
    shape: tuple[int, int]
    helper: MinesweeperHelper
    groups: AllGroups
    all_clusters: AllClusters

    def __init__(self, size: int):
        self.master = Tk()
        self.size = size
        self.reset()
        self.shape = (size, size)
        self.helper = MinesweeperHelper(self.shape)
        self.groups = AllGroups()
        self.all_clusters = None

    def reset(self) -> None:
        """
        Reset entire board
        :return:
        """
        self.mines = np.zeros((self.size, self.size)).astype(bool)
        self.buttons = np.zeros_like(self.mines).astype(Button)
        self.opened = np.zeros_like(self.mines).astype(bool)
        self.flagged = np.zeros_like(self.mines).astype(bool)
        self.set_start(-1, -1)
        self.mode = Mode.SET_MINES
        for i in range(self.size):
            for j in range(self.size):
                def btn_left(event):
                    x, y = getattr(event.widget, 'x2'), getattr(event.widget, 'y2')
                    if self.mode == Mode.SET_MINES:
                        self.toggle_mine(x, y)
                    elif self.mode == Mode.SET_START:
                        self.set_start(x, y)
                    elif self.mode == Mode.TRY:
                        self.toggle_cell_cover(x, y)

                def btn_middle(event):
                    x, y = getattr(event.widget, 'x2'), getattr(event.widget, 'y2')
                    if self.mode == Mode.SET_MINES:
                        pass
                    elif self.mode == Mode.SET_START:
                        pass
                    elif self.mode == Mode.TRY:
                        self.chord(x, y)

                def btn_right(event):
                    x, y = getattr(event.widget, 'x2'), getattr(event.widget, 'y2')
                    if self.mode == Mode.SET_MINES:
                        pass
                    elif self.mode == Mode.SET_START:
                        pass
                    elif self.mode == Mode.TRY:
                        self.toggle_flag(x, y)
                btn = Button(self.master, text='0', fg='white', bg='black', width=5, height=2, relief='flat', padx=2, pady=0)
                setattr(btn, 'x2', i)
                setattr(btn, 'y2', j)
                btn.bind(LEFT_MOUSE_BTN, btn_left)
                btn.bind(MIDDLE_MOUSE_BTN, btn_middle)
                btn.bind(RIGHT_MOUSE_BTN, btn_right)
                btn.grid(column=j, row=i, padx=0, pady=0, sticky=W)
                self.buttons[i][j] = btn

    def toggle_mine(self, x: int, y: int) -> None:
        """
        Toggle a mine at a given cell.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        self.mines[x, y] = not self.mines[x, y]
        self._redraw(x, y)

    def set_start(self, x: int, y: int) -> None:
        """
        Set the start field to a given cell.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        if x < 0 or y < 0:
            self.start_field = -1, -1
            return
        count = self._count_mines(x, y)
        if count != 0:
            print('Field is not a viable start field')
        else:
            x2, y2 = self.start_field
            if x2 >= 0 and y2 >= 0:
                self.buttons[x2, y2]['bg'] = COLORS[self._count_mines(x2, y2)]
                self.opened[x2, y2] = False
            self.start_field = x, y
            self.buttons[x, y]['bg'] = 'lawngreen'
            self.buttons[x, y]['fg'] = 'black'
            self.opened[x, y] = True

    def chord(self, x: int, y: int) -> None:
        """
        Initiate a chord on a given cell, uncovering neighbouring cells if the mine counts are already met.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        if not self.opened[x, y]:
            return
        count = self._count_mines(x, y)
        mask = self._get_neighbour_mask(x, y)
        flagged = self.flagged & mask
        possible_mines = (~self.opened & ~flagged) & mask
        count -= flagged.sum()
        leftovers_are_mines = count == possible_mines.sum()
        if leftovers_are_mines:
            # Flag remaining covered cells
            for x2 in self._get_range(x):
                for y2 in self._get_range(y):
                    if x == x2 and y == y2:
                        continue
                    if possible_mines[x2, y2]:
                        self.toggle_flag(x2, y2)
        elif count == 0:
            new_chords = []
            # Uncover all around
            for x2 in self._get_range(x):
                for y2 in self._get_range(y):
                    if x == x2 and y == y2:
                        continue
                    if not self.opened[x2, y2] and not self.flagged[x2, y2]:
                        self.toggle_cell_cover(x2, y2)
                        if self._count_mines(x2, y2) == 0:
                            new_chords.append((x2, y2))
            # Now recurse
            for x2, y2 in new_chords:
                self.chord(x2, y2)

    def toggle_cell_cover(self, x: int, y: int) -> None:
        """
        Toggles the cover of a cell.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        if self.flagged[x, y]:
            return
        self.opened[x, y] = ~self.opened[x, y]
        self._color_cell(x, y)

    def toggle_flag(self, x: int, y: int) -> None:
        """
        Toggles a flag, if the given cell is covered.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        if self.opened[x, y]:
            return
        self.flagged[x, y] = not self.flagged[x, y]
        self._color_cell(x, y)

    def export_board(self) -> str:
        """
        Export the current board configuration.
        :return: A string representation storing the board configuration.
        """
        s = ''
        n_mines = 0
        for x in range(self.size):
            s += '\n'
            for y in range(self.size):
                if self.mines[x, y]:
                    s += '||:boom:||'
                    n_mines += 1
                else:
                    count = self._count_mines(x, y)
                    if (x, y) == self.start_field:
                        s += f":{NUM2WORD[count]}:"
                    else:
                        s += f"||:{NUM2WORD[count]}:||"
        return f"{self.size}x{self.size} grid\n{n_mines} mines\n" + s

    def import_board(self, s: str):
        """
        Import a board configuration from a string.
        :param s: A string that holds the exported board configuration.
        :return:
        """
        lines = s.splitlines()
        if len(lines) <= 3:
            print("Cannot import grid: Clipboard too small")
            return
        match = re.match(r'(:?\d+)x(:?\d+) grid', lines[0])
        if match is None or len(match.groups()) != 2:
            print("Cannot import grid: Could not find header")
            return
        if match.group(1) != match.group(2) != self.size:
            print(f"Cannot import grid: Size {match.group(1)} is not equal to {self.size}")
            return
        for x, line in zip(range(self.size), lines[3:]):
            for y in range(self.size):
                if line.startswith(':zero:'):  # start cell
                    self.set_start(x, y)
                cell_match = re.match(r'^\|\|:(\w+):\|\||^:(\w+):', line, re.MULTILINE)
                if cell_match is None:
                    print(f'Error parsing cell from line at ({x}, {y}): "{line}"')
                    return
                if cell_match.group(1) is None:
                    cell_str = cell_match.group(2)
                else:
                    cell_str = cell_match.group(1)
                if cell_str == 'boom' and not self.mines[x][y]:
                    self.toggle_mine(x, y)
                elif cell_str in NUM2WORD.values() and self.mines[x][y]:
                    self.toggle_mine(x, y)
                line = line[len(cell_match.group(0)):]

    def restore_cover(self) -> None:
        """
        Cover the entire board, resetting a try.
        :return:
        """
        self.opened[:, :] = False
        self.flagged[:, :] = False
        self.set_start(*self.start_field)
        self._redraw()

    def _get_range(self, x_or_y: int) -> Iterable[int]:
        """
        Create the range in any direction given a position (either x or y).
        :param x_or_y: Either the x or the y coordinate.
        :return: An iterable range of the direct neighbours of that position.
        """
        return range(max(0, x_or_y - 1), min(self.size, x_or_y + 2))

    def _redraw(self, x: int = None, y: int = None) -> None:
        """
        Redraws the entire board or a given cell and all its neighbours.
        :param x: The x coordinate of the given cell (optional).
        :param y: The y coordinate of the given cell (optional).
        :return:
        """
        if x is None or y is None:
            range_x = range(self.size)
            range_y = range(self.size)
        else:
            range_x = self._get_range(x)
            range_y = self._get_range(y)
        for x2 in range_x:
            for y2 in range_y:
                self._color_cell(x2, y2)

    def cycle_mode(self) -> None:
        """
        Cycle through the different modes.
        :return:
        """
        if self.mode == Mode.SET_MINES:
            self.mode = Mode.SET_START
        elif self.mode == Mode.SET_START:
            self.mode = Mode.TRY
            self._redraw()
        elif self.mode == Mode.TRY:
            self.mode = Mode.SET_MINES
            self._redraw()

    def _color_cell(self, x: int, y: int) -> None:
        """
        Colors a given cell, considering the current state of the board.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        btn: Button = self.buttons[x][y]
        if self.mines[x][y]:
            self._color_mine(btn)
            if (x, y) == self.start_field:
                self.start_field = -1, -1
        else:
            count = self._count_mines(x, y)
            self._color_number(btn, count, x, y)
            if (x, y) == self.start_field:
                if count == 0:
                    btn['bg'] = 'lawngreen'
                else:
                    self.start_field = -1, -1
        if self.mode == Mode.TRY and not self.opened[x][y]:
            self._color_covered(btn, x, y)

    def _count_mines(self, x: int, y: int) -> int:
        """
        Counts the mines around a given cell.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return: The number of mines.
        """
        return (self._get_neighbour_mask(x, y) & self.mines).sum()

    def _get_neighbour_mask(self, x: int, y: int) -> np.ndarray:
        """
        Constructs a boolean mask that is True for all neighbours of the given cell coordinates.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return: A mask with the same dimensions as the board, that is True for all neighbours of the given cell.
        """
        arr = np.zeros_like(self.mines).astype(bool)
        xrange = list(self._get_range(x))
        yrange = list(self._get_range(y))
        arr[xrange[0]:xrange[-1] + 1, yrange[0]:yrange[-1] + 1] = True
        arr[x, y] = False
        return arr

    def _color_mine(self, btn: Button) -> None:
        """
        Colors in a button that represents a mine.
        :param btn: The button that represents a mine.
        :return:
        """
        btn['bg'] = 'red'
        btn['text'] = 'x'
        btn['fg'] = 'white'

    def _color_number(self, btn: Button, count: int, x: int, y: int) -> None:
        """
        Colors in a button that shows a mine proximity number.
        :param btn: The button that shows a mine proximity number.
        :param count: The number of neighbouring mines to display on the button.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        btn['bg'] = COLORS[count]
        btn['text'] = count
        if count <= 1 and (x, y) != self.start_field:
            btn['fg'] = 'white'
        else:
            btn['fg'] = 'black'

    def _color_covered(self, btn: Button, x: int, y: int) -> None:
        """
        Colors in a button that is covered.
        Could show a flag on top if marked.
        :param btn: The button that represents a covered cell.
        :param x: The x coordinate of the given cell.
        :param y: The y coordinate of the given cell.
        :return:
        """
        btn['bg'] = 'black'
        btn['fg'] = 'black'
        if self.flagged[x, y]:
            btn['text'] = '🚩'
        else:
            btn['text'] = ''

    def solve(self) -> None:
        """
        Solve the board from the current state.
        :return:
        """


        # https://git.tartarus.org/?p=simon/puzzles.git;a=blob;f=mines.c;h=d8233417d9d540dfde618d6a7052f868a7345005;hb=HEAD#l647
        cells_to_process = self.opened.copy()
        while self.solve_step(cells_to_process) and cells_to_process.sum() != 0:
            pass
        return self.opened.sum() == self.size ** 2

        # if self.start_field == (-1, -1):
        #     return
        # self.restore_cover()
        # self.chord(*self.start_field)
        # constraints = np.zeros_like(self.mines)
        # counts = np.zeros_like(self.mines).astype(bool)
        # for x, y in np.ndindex(constraints.shape):
        #     constraints[x, y] = np.ndarray(np.transpose(self._get_neighbour_mask(x, y).nonzero()))
        #     counts[x, y] = self._count_mines(x, y) if not self.opened[x, y] else False
        # return True

    def method_naive(self):
        """ Method #1. Naive.
        Try to find safe and mines in the groups themselves.
        Return safe and mines found in those groups
        """
        safe, mines = [], []
        for group in self.groups:

            # No remaining mines
            if group.is_all_safe():
                safe.extend(list(group.cells))

            # All covered are mines
            if group.is_all_mines():
                mines.extend(list(group.cells))

        return list(set(safe)), list(set(mines))

    def method_groups(self):
        """ Method #2. Groups.
        Cross check all groups. When group is a subset of
        another group, try to deduce safe cells and mines.
        """
        safe, mines = [], []

        # Cross-check all-with-all groups
        for group_a in self.groups:
            for group_b in self.groups:

                # Don't compare with itself
                if group_a.hash == group_b.hash:
                    continue

                safe.extend(self.deduce_safe(group_a, group_b))
                mines.extend(self.deduce_mines(group_a, group_b))

                # If group A is a subset of group B and B has more mines
                # we can create a new group that would contain
                # B-A cells and B-A mines
                # len(group_b.cells) < 8 prevents computational explosion on
                # multidimensional fields
                if len(group_b.cells) < 8 and \
                   group_a.cells.issubset(group_b.cells) and \
                   group_b.mines - group_a.mines > 0:
                    new_group = mc.MineGroup(group_b.cells - group_a.cells,
                                             group_b.mines - group_a.mines)
                    self.groups.add_group(new_group)

        return list(set(safe)), list(set(mines))

    def method_subgroups(self):
        """ Method #3. Subgroups. Breaking down groups into "subgroups":
        "at least" and "no more than". Cross-check them with regular groups
        to deduce mines.
        """
        # Note how many groups we have
        self.groups.count_groups = len(self.groups.mine_groups)

        # Generate subgroups
        # Funny thing, it actually works just as well with only one
        # (either) of these two generated
        self.groups.generate_subgroup_at_least()
        self.groups.generate_subgroup_no_more_than()

        safe, mines = [], []

        # The idea is similar to the "groups" method:
        # cross-check all the groups, but this time
        # we only will check "at least" and "no more than"
        # subgroups
        # Group A are all subgroups (at least, no more)
        for group_a in self.groups.subgroups():
            # Group B are all groups (exactly)
            for group_b in self.groups.exact_groups():

                # Only compare subgroups "at least" to groups.
                if group_a.group_type == "at least":

                    # Similar to "groups" method: if mines are the same,
                    # the difference is safe
                    # Subgroup A (cells 1, 2) has at least X mines,
                    # Group B (1, 2, 3) has X mines: then cell3 is safe
                    safe.extend(self.deduce_safe(group_a, group_b))

                # Only compare subgroups "no more than" to groups.
                if group_a.group_type == "no more than":

                    # Similar to "groups" method: if mines are the same,
                    # the difference is safe
                    # Subgroup A (cells 1, 2) has at least X mines,
                    # Group B (1, 2, 3) has X mines: then cell3 is safe
                    mines.extend(self.deduce_mines(group_a, group_b))

        return list(set(safe)), list(set(mines))

    def method_csp(self):
        """ Method #4. CSP (Constraint Satisfaction Problem).
        Generate overlapping groups (clusters). For each cluster find safe
        cells and mines by brute forcing all possible solutions.
        """
        safe, mines = [], []

        # Generate clusters
        self.generate_clusters()
        # Do all the solving / calculate frequencies stuff
        self.all_clusters.calculate_all()

        for cluster in self.all_clusters.clusters:

            # Get safe cells and mines from cluster
            safe.extend(cluster.safe_cells())
            mines.extend(cluster.mine_cells())

        return list(set(safe)), list(set(mines))

    def method_coverage(self):
        """ Method #5, Coverage.
        Deduce safes and  mines from the "unaccounted" group
        """
        # Trivial coverage cases: no mines and all mines
        if self.remaining_mines == 0:
            return self.covered_cells, []
        if len(self.covered_cells) == self.remaining_mines:
            return [], self.covered_cells

        if self.unaccounted_group is None:
            return [], []

        safe, mines = [], []
        if self.unaccounted_group.is_all_safe():
            safe.extend(list(self.unaccounted_group.cells))
        if self.unaccounted_group.is_all_mines():
            mines.extend(list(self.unaccounted_group.cells))
        return list(set(safe)), list(set(mines))

    def method_bruteforce(self):
        """Bruteforce mine probabilities, when there are only a handful
        of cells/mines left. This replaces more sophisticate
        calculate_probabilities method.
        """
        # Use this method only if there is not a lot combinations to go through
        if len(self.covered_cells) > 25 or \
           math.comb(len(self.covered_cells), self.remaining_mines) > 3060:
            return [], []

        safe, mines = [], []
        self.generate_bruteforce()

        # Go through all cells
        for position, cell in enumerate(self.covered_cells):
            # And count mines in all solutions in this position
            solutions_with_mines = 0
            for solution in self.bruteforce_solutions:
                if solution[position]:
                    solutions_with_mines += 1

            # If there were no mines - this cell is safe
            if solutions_with_mines == 0:
                safe.append(cell)
            # If there were as many mines as solutions - it's a mine
            elif solutions_with_mines == len(self.bruteforce_solutions):
                mines.append(cell)

        return safe, mines

    def bruteforce_probabilities(self):
        """Bruteforce mine probabilities, when there are only a handful
        of cells/mines left. This replaces more sophisticate
        calculate_probabilities method.
        """

    def calculate_probabilities(self):
        """ Final method. "Probability". Use various methods to determine
        which cell(s) is least likely to have a mine
        """

        def background_probabilities(self):
            """ Populate self.probabilities based on background probability.
            Which is All mines divided by all covered cells.
            It is quite crude and often inaccurate, it is just a fallback
            if any of more accurate methods don't work.
            """
            background_probability = \
                self.remaining_mines / len(self.covered_cells)

            for cell in self.covered_cells:
                self.probability.cells[cell] = \
                    mc.CellProbability(cell, "Background",
                                       background_probability)

        def probabilities_for_groups(self):
            """ Update self.probabilities, based on mine groups.
            For each group consider mine probability as "number of mines
            divided by the number of cells".
            """
            for group in self.groups.exact_groups():

                # Probability of each mine in the group
                group_probability = group.mines / len(group.cells)
                for cell in group.cells:

                    # If group's probability is higher than the background:
                    # Overwrite the probability result
                    if group_probability > \
                       self.probability.cells[cell].mine_chance:
                        self.probability.cells[cell] = \
                            mc.CellProbability(cell, "Groups",
                                               group_probability)

        def csp_probabilities(self):
            """ Update self.probabilities based on results from CSP method.
            """
            for cluster in self.all_clusters.clusters:
                for cell, frequency in cluster.frequencies.items():
                    # Overwrite the probability result
                    self.probability.cells[cell] = \
                        mc.CellProbability(cell, "CSP", frequency)

        def cluster_leftovers_probabilities(self):
            """ Update self.probabilities based on "leftovers",
            cells that are not in any clusters. (not to be confused with
            "Unaccounted" - those are cells that are not in any group).
            """
            self.all_clusters.calculate_leftovers()

            # For some reasons, calculation failed
            # (probably clusters were too long, or unsolvable)
            if self.all_clusters.leftover_mine_chance is None:
                return

            # Fill in the probabilities
            for cell in self.all_clusters.leftover_cells:
                self.probability.cells[cell] = \
                    mc.CellProbability(cell, "CSP Leftovers",
                                       self.all_clusters.leftover_mine_chance)

        # Reset probabilities
        self.probability = mc.AllProbabilities()
        # Background probability: all remaining mines on all covered cells
        background_probabilities(self)
        # Based on mines in groups
        probabilities_for_groups(self)
        # Based on CSP solutions
        csp_probabilities(self)
        # Probabilities of non-cluster mines
        cluster_leftovers_probabilities(self)

    def calculate_opening_chances(self):
        """ Populate opening_chance in self.probabilities: a chance that this
        cell is a zero. (Which is a good thing)
        """
        # Go through all cells we have probability info for
        # (that would be all covered cells)
        for cell, cell_info in self.probability.cells.items():
            zero_chance = 1
            # Look at neighbors of each cell
            for neighbor in self.helper.cell_surroundings(cell):
                # If there are any mines around, there is no chance of opening
                if self.field[neighbor] == mg.CELL_MINE:
                    cell_info.opening_chance = 0
                    break
                # Otherwise each mine chance decrease opening chance
                # by (1 - mine chance) times
                if neighbor in self.probability.cells:
                    zero_chance *= \
                        (1 - self.probability.cells[neighbor].mine_chance)
            else:
                self.probability.cells[cell].opening_chance = zero_chance

    def calculate_frontier(self):
        """ Populate frontier (how many groups may be affected by this cell)
        """
        # Generate frontier
        self.groups.generate_frontier()

        for cell in self.groups.frontier:
            for neighbors in self.helper.cell_surroundings(cell):
                if neighbors in self.probability.cells:
                    self.probability.cells[neighbors].frontier += 1

    def calculate_next_safe_csp(self):
        """ Populate "next safe" information (how many guaranteed safe cells
        will be in the next move, based on CSP solutions).
        """
        # Do the calculations
        self.all_clusters.calculate_all_next_safe()

        # Populate probability object with this info
        for cluster in self.all_clusters.clusters:
            for cell, next_safe in cluster.next_safe.items():
                self.probability.cells[cell].csp_next_safe = next_safe

    @staticmethod
    def pick_a_random_cell(cells):
        """Pick a random cell out of the list of cells.
        (Either for testing or when we are reduced to guessing from list
        of cells with exactly the same probabilities)
        """
        return random.choice(cells)

    def solve_step(self, deterministic=False, optimize_for_speed=False, this_is_next_move=False):
        ''' Main solving function.
        Go through various solving methods and return safe and mines lists
        as long as any of the methods return results
        In:
        - the field (what has been uncovered so far).
        - deterministic: don't use random at all. In case of several equally
          probably safe cells, pick the first one
        - optimize_for_speed: use faster but a bit less winning settings
          (no recursion into the next move)
        - this_is_next_move: Flag that this run is a recursive run for the next move

        Out:
        - list of safe cells
        - list of mines
        '''
        if optimize_for_speed:
            this_is_next_move = False

        # First click on the "all 0" corner
        if self.opened.sum() == self.opened.shape[0] * self.opened.shape[1]:
            self.last_move_info = ("First click", None, None)
            all_zeros = tuple(0 for _ in range(len(self.shape)))
            return [all_zeros, ], None

        # Several calculation needed for the following solution methods
        # A list of all covered cells
        # self.generate_all_covered()
        # Number of remaining mines
        # self.calculate_remaining_mines()
        # Generate groups (main data for basic solving methods)
        self.generate_groups()
        # Unaccounted cells (covered minus mines, has  to go after the  groups)
        self.generate_unaccounted()

        # These are 6 deterministic methods to try
        solution_methods = [
            (self.method_naive, "Naive"),
            (self.method_groups, "Groups"),
            (self.method_subgroups, "Subgroups"),
            (self.method_coverage, "Coverage"),
            (self.method_csp, "CSP"),
            (self.method_bruteforce, "Bruteforce"),
        ]

        if optimize_for_speed:
            solution_methods = solution_methods[:5]
        elif this_is_next_move:
            solution_methods = solution_methods[:4]

        # If any of the methods returned results - return the result
        for method, method_name in solution_methods:
            # Run the method from the list
            safe, mines = method()
            # If method was successful (found at least safe or mine)
            if safe or mines:
                # Update the last move info and return found cells
                self.last_move_info = (method_name, None, None)
                return safe, mines

        # Empty bruteforce means illegal field (may happen when we try
        # all combinations for the next move)
        if self.bruteforce_solutions == []:
            return [-1], [-1]

        # Calculate mine probability using various methods
        self.calculate_probabilities()
        # Calculate safe cells for the next move in CSP
        self.calculate_next_safe_csp()

        # Two more calculations that will be used to pick
        # the best random cell:
        # Opening chances (chance that cell is a zero)
        self.calculate_opening_chances()
        # Does it touch a frontier (cells that already are in groups)
        self.calculate_frontier()

        # Get cells that is least likely a mine
        next_moves = 1
        # If we optimize for speed or this is already "next_move"
        # set next_moves to 0
        if optimize_for_speed or this_is_next_move:
            next_moves = 0
        lucky_cells = \
            self.probability.get_luckiest(self.all_clusters, next_moves,
                                          deterministic, self)

        if lucky_cells:

            # There may be more than one such cells, so either
            # Pick the fist one, if solver is in deterministic mode
            if deterministic:
                lucky_cell = lucky_cells[0]
            # or pick a random one
            else:
                lucky_cell = self.pick_a_random_cell(lucky_cells)


            # Store information about expected chance of mine and how
            # this chance was calculated
            self.last_move_info = \
                ("Probability",
                 self.probability.cells[lucky_cell].source,
                 self.probability.cells[lucky_cell].mine_chance)
            return [lucky_cell, ], None

        # This should not happen, but here's a catch-all if it does
        self.last_move_info = ("Last Resort", None, None)
        return [self.pick_a_random_cell(self.covered_cells), ], None

    def generate_groups(self):
        """ Populate self.group with MineGroup objects
        """

        # Reset the groups
        self.groups.reset()

        # Go over all cells and find all the "Numbered ones"
        for cell in self.helper.iterate_over_all_cells():

            # Groups are only for numbered cells
            if not self.opened[cell]:
                continue

            # For them we'll need to know two things:
            # What are the uncovered cells around it
            covered_neighbors = []
            # And how many "Active" (that is, minus marked)
            # mines are still there
            active_mines = self.mines.sum() - self.flagged.sum()

            # Go through the neighbors
            for neighbor in self.helper.cell_surroundings(cell):
                # Collect all covered cells
                if not self.opened[neighbor]:
                    covered_neighbors.append(neighbor)
                # Subtract all marked mines
                if self.mines[neighbor]:
                    active_mines -= 1

            # If the list of covered cells is not empty:
            # store it in the self.groups
            if covered_neighbors:
                new_group = MineGroup(covered_neighbors, active_mines)
                self.groups.add_group(new_group)
