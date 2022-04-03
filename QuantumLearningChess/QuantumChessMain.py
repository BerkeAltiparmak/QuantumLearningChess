import pandas as pd
import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from ChessEngine import GameState, Move


def get_specific_dataframe(theme1: str = 'endgame', theme2: str = 'mate',
                           file_name: str = 'lichess_db_puzzle.csv') -> pd.DataFrame:
    """
    Return a dataframe that consists of puzzles with the given two themes and the chess puzzle
     database file.

    :param theme1: one of the chess themes the user wants from the chess database
    :param theme2: the other chess theme the user wants from the chess database
    :param file_name: the name and the location of the chess database file
    :return: a specific dataframe that consists of chess database with theme1 and theme2
    """
    df = pd.read_csv(file_name)
    df.columns = ['PuzzleId', 'FEN', 'Moves', 'Rating', 'RatingDeviation', 'Popularity', 'NbPlays', 'Themes', 'GameUrl']
    specific_df = df.loc[df['Themes'].str.contains(theme1, na=False) & df['Themes'].str.contains(theme2, na=False)]
    return specific_df


def preprocess_data(df: pd.DataFrame, size: int, train_ratio: float = 0.8) \
        -> (list[list[list[tuple]]], list[list[list[tuple]]]):
    """
    Return an 8x8 pixel version of each FEN (a FEN is a standard notation for describing a
    particular board position of a chess game) in the chess puzzle data frame, where each pixel
    represents the piece in that position on the board given by the FEN.

    :param df: the specific or nonspecific chess puzzle dataframe.
    :param size: the maximum size we want the returned list to be.
    :param train_ratio: train_size / df.size; the ratio of the size of the train data
    :return: 8x8 pixel of the chess board, where each pixel represents a piece of each FEN in df.
    """
    size = size if size < df.size else df.size  # to ensure the size is less than size of df
    converted_data = []

    # for each FEN in the chess puzzle database, first create a GameState object so that it follows
    # the rules of chess. Then, convert the board of a GameState object into a 8x8 pixel, where
    # each pixel is one-hot encoded version of a piece on the board.
    for i in range(0, size):
        FEN = df.iloc[i]['FEN']
        moves = df.iloc[i]['Moves'].split(' ')
        game_state = GameState()
        game_state.FEN_to_board(FEN)  # convert Lichess puzzle FEN into a GameState object
        # Lichess puzzle occurs after the first move, so we have to execute this code:
        game_state.make_move(Move([ord(moves[0][0]) - 97, int(moves[0][1]) - 1],
                                  [ord(moves[0][2]) - 97, int(moves[0][3]) - 1], game_state.board))
        if game_state.white_to_move:  # for now, our model will only train for puzzles for white
            converted_data.append(board_to_pixels(game_state.board))  # pixelated chess board

    train_data = converted_data[:round(len(converted_data) * train_ratio)]
    test_data = converted_data[len(train_data):]
    return (train_data, test_data)


def board_to_pixels(board: list[list[str]]) -> list[list[tuple]]:
    """
    Given the chess board, one-hot encode all of the pieces so that the machine learning algorithm
    is not biased. After encoding, return the board as an image; that is, every piece (or an empty
    square) is a pixel on the board, and the board is an 8x8 pixel.
    :param board: position of the board of a GameState object.
    :return: pixelated (converted into a 8x8 pixel image) version of the boardç
    """
    piece_to_tuple = {'wp': (1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'bp': (0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0),
                      'wN': (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'bN': (0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0),
                      'wB': (0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0), 'bB': (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
                      'wR': (0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0), 'bR': (0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0),
                      'wQ': (0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0), 'bQ': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0),
                      'wK': (0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0), 'bK': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1),
                      '--': (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)}
    pixelated_board = []
    for row in board:
        pixelated_row = []
        for piece in row:
            pixelated_row.append(piece_to_tuple[piece])
        pixelated_board.append(pixelated_row)
    return pixelated_board


def quantum_preprocessing(train_chess_data: list[list[list[tuple]]],
                          test_chess_data: list[list[list[tuple]]],
                          qua_preprocess: bool = True):
    n_epochs = 50  # Number of optimization epochs
    n_layers = 1  # Number of random layers

    SAVE_PATH = "quantum_chess_learning/"  # Data saving folder
    np.random.seed(0)  # Seed for NumPy random number generator
    tf.random.set_seed(0)  # Seed for TensorFlow random number generator

    # Add extra dimension for convolution channels
    train_chess_data = np.array(train_chess_data[:, tf.newaxis], requires_grad=False)
    test_chess_data = np.array(test_chess_data[:, tf.newaxis], requires_grad=False)

    # Setting up the quantum machine learning device with 4 wires
    dev = qml.device("default.qubit", wires=4)
    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

    @qml.qnode(dev)
    def circuit(phi):
        # Encoding of 4 classical input values
        for j in range(4):
            qml.RY(np.pi * phi[j], wires=j)

        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(4)))

        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]

    if qua_preprocess == True:
        q_train_images = []
        print("Quantum pre-processing of train images:")
        for idx, img in enumerate(train_chess_data):
            print("{}/{}        ".format(idx + 1, n_train), end="\r")
            q_train_images.append(quanv(img))
        q_train_images = np.asarray(q_train_images)

        q_test_images = []
        print("\nQuantum pre-processing of test images:")
        for idx, img in enumerate(test_chess_data):
            print("{}/{}        ".format(idx + 1, n_test), end="\r")
            q_test_images.append(quanv(img))
        q_test_images = np.asarray(q_test_images)

        # Save pre-processed images
        np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
        np.save(SAVE_PATH + "q_test_images.npy", q_test_images)

    # Load pre-processed images
    q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
    q_test_images = np.load(SAVE_PATH + "q_test_images.npy")


if __name__ == '__main__':
    df = get_specific_dataframe()
    train_chess_data, test_chess_data = preprocess_data(df, 10000)
