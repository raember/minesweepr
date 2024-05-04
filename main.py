import argparse
from subprocess import Popen, PIPE
from tkinter import mainloop, Button

from board import Board, Mode


def export_board():
    global board
    p = Popen(['xsel', '-ib'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate(board.export_board().encode('utf-8'))
    if len(err) > 0:
        print(f"Failed to set clipboard: {err}")


def import_board():
    p = Popen(['xsel', '-b'], stdin=PIPE, stdout=PIPE, stderr=PIPE)
    out, err = p.communicate()
    if len(err) > 0:
        print(f"Failed to set clipboard: {err}")
    global board
    board.import_board(out.decode())


def create():
    pass


def solve():
    global board
    board.solve()


def toggle_mode():
    global board
    board.cycle_mode()
    if board.mode == Mode.SET_MINES:
        mode_btn['bg'] = 'darkred'
        mode_btn['fg'] = 'black'
        mode_btn['text'] = 'mine'
    elif board.mode == Mode.SET_START:
        mode_btn['bg'] = 'limegreen'
        mode_btn['fg'] = 'white'
        mode_btn['text'] = 'start'
    elif board.mode == Mode.TRY:
        mode_btn['bg'] = 'deepskyblue'
        mode_btn['fg'] = 'black'
        mode_btn['text'] = 'try'


def restore_cover():
    global board
    board.restore_cover()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('minesweepr')
    parser.add_argument('SIZE', action='store', type=int,
                        help='The size of the mine field.')
    args = parser.parse_args()

    board = Board(args.SIZE)
    master = board.master
    mode_btn = Button(master, text='mine', relief='flat', command=toggle_mode, bg='darkred')
    mode_btn.grid(row=args.SIZE + 1, column=0, columnspan=args.SIZE // 2, padx=1, pady=4)
    cover_btn = Button(master, text='cover', relief='flat', command=restore_cover)
    cover_btn.grid(row=args.SIZE + 2, column=0, columnspan=args.SIZE // 2, padx=1, pady=4)
    create_btn = Button(master, text='create', relief='flat', command=create)
    create_btn.grid(row=args.SIZE + 3, column=0, columnspan=args.SIZE // 2, padx=1, pady=4)
    export_btn = Button(master, text='export', relief='flat', command=export_board)
    export_btn.grid(row=args.SIZE + 1, column=args.SIZE // 2 + 1, columnspan=args.SIZE, padx=1, pady=4)
    import_btn = Button(master, text='import', relief='flat', command=import_board)
    import_btn.grid(row=args.SIZE + 2, column=args.SIZE // 2 + 1, columnspan=args.SIZE, padx=1, pady=4)
    solve_btn = Button(master, text='solve', relief='flat', command=solve)
    solve_btn.grid(row=args.SIZE + 3, column=args.SIZE // 2 + 1, columnspan=args.SIZE, padx=1, pady=4)
    mainloop()
