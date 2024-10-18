import pandas as pd
import numpy as np

def parse_file_to_dataframe(file_path, delimiter=r'\s+'):
    """
    Parses a file with numerical data into a pandas DataFrame.

    :param file_path: Path to the file to be parsed.
    :param delimiter: Regular expression used to separate values in the file.
                      Default is a regular expression matching one or more spaces (r'\s+').
    :return: DataFrame with the parsed data.
    """
    try:
        # Read the file into a DataFrame
        df = pd.read_csv(file_path, sep=delimiter, header=None)

        # Assign column names based on the number of columns detected
        num_columns = df.shape[1]
        column_names = [f'Column_{i+1}' for i in range(num_columns)]
        df.columns = column_names

        return df

    except Exception as e:
        print(f"Error parsing the file: {e}")
        return None

def get_meta_data():
    """
    Processes metadata from a file to create a filtered DataFrame.

    https://webdav.tuebingen.mpg.de/cause-effect/README

    Returns:
        pandas.DataFrame: A DataFrame with processed metadata.
    """
    # Parse the file into a DataFrame
    meta_df = parse_file_to_dataframe("data/pairmeta.txt")

    # Filter rows where all selected columns have values 1 or 2
    meta_filter = ((meta_df.iloc[:, 1:5] == 1) | (meta_df.iloc[:, 1:5] == 2)).all(axis=1)
    df = meta_df[meta_filter].copy()

    # Determine direction based on specific column values and add as a new column
    df['direction'] = np.where(
        (df['Column_2'] == 1) & (df['Column_3'] == 1) &
        (df['Column_4'] == 2) & (df['Column_5'] == 2),
        1, -1
    )

    # Drop Columns 2 to 5 as they are no longer needed
    df = df.drop(columns=[f"Column_{i}" for i in range(2, 6)])

    # Rename columns for clarity
    df = df.rename(columns={"Column_1": "file #", "Column_6": "dataset weight"})

    return df


if __name__ == "__main__":

    df_meta = get_meta_data()

    scalar_df_list = []
    for i, d in zip(df_meta["file #"], df_meta["direction"]):
        file_path = f"data/pair{str(i).zfill(4)}.txt"
        df_i   = parse_file_to_dataframe(file_path)

        # pick examples that contain only two scalar variables
        if len(df_i.columns) > 2:
            continue
        elif df_i.isna().any().any():
            df_cleaned = df.dropna()
            print(f"NaNs dropped in ", file_path)

        scalar_df_list.append((i, df_i, d))


    print("# of data sets = ", len(scalar_df_list))
    print(df_i)
