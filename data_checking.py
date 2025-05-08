import numpy as np
import pandas as pd
from pathlib import Path

from pandas import DataFrame

from colorama import Fore
from matplotlib import pyplot as plt
import mplcyberpunk


def create_plot_many(rows: int, columns: int, subtitle: str, df: DataFrame, features: list = None):
    fig, axs = plt.subplots(rows, columns, figsize=(10, 8), sharex=True)  # Share x-axis for better alignment
    fig.suptitle(f'Data View of {subtitle}', fontsize=16, fontweight='bold')

    for i, feature in enumerate(features):
        if len(df.index) > 1:
            axs[i].plot(df[feature], label=feature, linewidth=2, alpha=0.8, color=line_colors[i])
        else:
            axs[i].scatter(df.index, df[feature], label=feature, linewidth=2, alpha=0.8, color=line_colors[i])
        axs[i].set_ylabel(feature, fontsize=12)
        axs[i].legend(loc='upper right')
        axs[i].grid(True, linestyle='--', alpha=0.6)
        try:
            mplcyberpunk.make_lines_glow(axs[i])
            mplcyberpunk.add_underglow(axs[i])
            mplcyberpunk.add_gradient_fill(axs[i], alpha_gradientglow=0.3)
        except:
            print(f"mplcyberpunk error")
    # X-axis label (only for the last subplot)
    axs[-1].set_xlabel("Time (Index)", fontsize=12)
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leaves space for the title
    return fig, axs


def create_plot_one():
    pass


def cyberpunk(axs):
    mplcyberpunk.make_lines_glow(axs)
    mplcyberpunk.add_underglow(axs)
    mplcyberpunk.add_gradient_fill(axs, alpha_gradientglow=0.3)

plt.style.use("cyberpunk")
# mplcyberpunk.add_glow_effects()
# mplcyberpunk.make_lines_glow()
# mplcyberpunk.add_underglow()
# mplcyberpunk.add_gradient_fill(alpha_gradientglow=0.5)
line_colors = ["#FF4500", "#1E90FF", "#DA70D6", "#00CED1"]
lighter_colors = ["#FF9999","#99CCFF","#99FF99","#FFCC99"]


def function1():
    save_path = Path("dane/graphical_presentation_old/original_data")
    features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']
    path = Path("dane/humidity_ofset_fixed")
    for file in path.glob("*.csv"):
        # if not "22" in file.name:
        #     continue
        df = pd.read_csv(file)

        # Create subplots (4 rows, 1 column)
        fig, axs = plt.subplots(4, 1, figsize=(10, 8), sharex=True)  # Share x-axis for better alignment
        fig.suptitle(f'Data View {file.name.split(".")[0]}', fontsize=16, fontweight='bold')

        # Plot each feature in a separate subplot
        for i, feature in enumerate(features):
            axs[i].plot(df[feature], label=feature, linewidth=2, alpha=0.8, color=line_colors[i])
            axs[i].set_ylabel(feature, fontsize=12)
            axs[i].legend(loc='upper right')
            axs[i].grid(True, linestyle='--', alpha=0.6)
            mplcyberpunk.make_lines_glow(axs[i])
            mplcyberpunk.add_underglow(axs[i])
            mplcyberpunk.add_gradient_fill(axs[i], alpha_gradientglow=0.3)


        # X-axis label (only for the last subplot)
        axs[-1].set_xlabel("Time (Index)", fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leaves space for the title
        # plt.savefig(save_path / f'{file.name.split(".")[0]}.png', dpi=300, bbox_inches="tight")
    # # Show the plot
    plt.show()

def function2():
    base_dir_path = Path("dane")
    folders = ["normalized_data", "generated_data"]
    for folder in folders:
        print(f"{Fore.LIGHTBLUE_EX}{folder}{Fore.RESET}")
        folder_path = base_dir_path / folder

        if folder_path.exists():
            for file in folder_path.glob("*.csv"):
                print(f"{Fore.LIGHTMAGENTA_EX}{file}{Fore.RESET}")

                df = pd.read_csv(file)

                total_records = len(df)

                label_counts = df['label'].value_counts()
                label_0 = label_counts.get(0, 0)
                label_1 = label_counts.get(1, 0)

                # Calculate proportions
                if total_records > 0:
                    prop_0 = label_0 / total_records * 100
                    prop_1 = label_1 / total_records * 100
                else:
                    prop_0 = prop_1 = 0

                print(f"File: {file}")
                print(f" - Total records: {total_records}")
                print(f" - Label 0 count: {label_0} ({prop_0:.2f}%)")
                print(f" - Label 1 count: {label_1} ({prop_1:.2f}%)")

def function3():
    save_path = Path("dane/graphical_presentation_old/generated_data")
    features = ['value_temp', 'value_hum', 'value_acid']
    path = Path("dane") / "generated_data"
    for file in path.glob("*.csv"):
        df = pd.read_csv(file)
        print(file.name)
        # Create subplots (3 rows, 1 column)
        fig, axs = plt.subplots(len(features), 1, figsize=(10, 8), sharex=True)  # Share x-axis for better alignment
        fig.suptitle(f'Data View {file.name.split(".")[0]}', fontsize=16, fontweight='bold')

        # Plot each feature in a separate subplot
        for i, feature in enumerate(features):
            axs[i].plot(df[feature], label=feature, linewidth=2, alpha=0.8, color=line_colors[i])
            axs[i].set_ylabel(feature, fontsize=12)
            axs[i].legend(loc='upper right')
            axs[i].grid(True, linestyle='--', alpha=0.6)
            mplcyberpunk.make_lines_glow(axs[i])
            mplcyberpunk.add_underglow(axs[i])
            mplcyberpunk.add_gradient_fill(axs[i], alpha_gradientglow=0.3)


        # X-axis label (only for the last subplot)
        axs[-1].set_xlabel("Time (Index)", fontsize=12)

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Leaves space for the title
        plt.savefig(save_path / f'{file.name.split(".")[0]}.png', dpi=300, bbox_inches="tight")
        plt.close()

def function4():
    """
    function that extends time gaps with nand values
    :return:
    """
    # Load dataset
    file_name = "df_RuralIoT_001"
    df = pd.read_csv(f"dane/{file_name}.csv", parse_dates=["time"])

    # Sort data by time
    df = df.sort_values("time").reset_index(drop=True)

    # Compute time difference between consecutive readings
    df["time_diff"] = df["time"].diff().dt.total_seconds()


    # Detect large gaps (threshold: 3x the median time difference)
    gap_threshold = 3 * df["time_diff"].median()
    df["large_gap"] = df["time_diff"] > gap_threshold

    # Assign segment IDs (each new gap starts a new segment)
    df["segment"] = df["large_gap"].cumsum()

    # Store resampled segments
    resampled_dfs = []

    # Process each segment separately
    for segment_id, segment_data in df.groupby("segment"):
        if len(segment_data) < 2:  # Skip segments with only one data point
            continue

        # Compute segment-specific mean interval
        segment_mean_interval = segment_data["time_diff"].mean()
        print(f"Segment {segment_id}: Mean interval = {segment_mean_interval:.2f} sec")

        # Resample using segment-specific interval (without filling gaps)
        segment_data.set_index("time", inplace=True)
        segment_resampled = segment_data.resample(f"{int(segment_mean_interval)}S").asfreq()

        # Keep segment_id column for tracking
        segment_resampled["segment"] = segment_id

        # Append to list
        resampled_dfs.append(segment_resampled)

    # Concatenate all resampled segments
    final_resampled_df = pd.concat(resampled_dfs).reset_index()

    # Save to CSV
    output_file = Path(f"dane/original_data_extended/{file_name}_extended.csv")
    final_resampled_df.to_csv(output_file, index=False)

    print(f"Resampled data saved to: {output_file}")

    pass

def segment_creator(data_path: Path, save_path: Path, large_gap_threshold: int):

    paths = data_path.glob("*.csv")

    for path in paths:
        print(f"processing {path.name}")
        segmentor(path, save_path, large_gap_threshold)



def segmentor(path: Path, save_path, large_gap_threshold):
    # Load dataset
    # path = Path("dane") / "df_RuralIoT_001.csv"
    df = pd.read_csv(path, parse_dates=["time"])

    # Sort data by time
    df = df.sort_values("time").reset_index(drop=True)


    # Compute time difference between consecutive readings
    df["time_diff"] = df["time"].diff().dt.total_seconds()


    median_time_interval = df["time_diff"].median()

    print(f"{median_time_interval=}")
    # large_gap_threshold = 4 * 60 * 60 # 4 hours
    df["medium_gap"] = (df["time_diff"] > median_time_interval*2) & (df["time_diff"] <= large_gap_threshold)
    df["large_gap"] = df["time_diff"] > large_gap_threshold


    # Assign segment IDs (each new large gap starts a new file)
    df["segment"] = df["large_gap"].cumsum()

    df["artificial"] = False
    # Directory to save processed CSVs
    output_dir = save_path / f"{path.stem}"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = df.drop(columns=["Unnamed: 0"])
    # Process each segment separately
    for segment_id, segment_data in df.groupby("segment"):
        # if len(segment_data) < 20:  # Rule 1: Skip short sequences
        #     print(segment_id, " skipped")
        #     continue

        # Create a new dataframe to store the resampled segment
        segment_resampled = segment_data.copy()

        # Add dummy timestamps for medium gaps
        new_rows = []
        for i in range(1, len(segment_data)):
            if segment_data.iloc[i]["medium_gap"]:  # If it's a medium gap
                start_time = segment_data.iloc[i - 1]["time"]
                end_time = segment_data.iloc[i]["time"]
                num_missing = int((end_time - start_time).total_seconds() / median_time_interval)

                # Generate missing timestamps with NaN values
                for j in range(1, num_missing):
                    missing_time = start_time + pd.Timedelta(seconds=j * median_time_interval)
                    new_rows.append([missing_time, np.NAN, np.nan, np.nan, np.nan, median_time_interval, np.nan, np.nan, segment_id, True])  # Add NaNs

        # Append missing rows to segment
        if new_rows:
            missing_df = pd.DataFrame(new_rows, columns=df.columns)
            segment_resampled = pd.concat([segment_resampled, missing_df]).sort_values("time").reset_index(
                drop=True)

        segment_resampled.set_index("time", inplace=True)

        # Save to CSV
        output_file = output_dir / f"segment_{segment_id}.csv"
        segment_resampled.to_csv(output_file, na_rep=np.nan)
        print(f"Saved to: {output_file}")


def function6():
    # Load processed CSV file (adjust the filename as needed)
    file_path = Path("dane/processed_data/segment_7_df_RuralIoT_001.csv")
    df = pd.read_csv(file_path, parse_dates=["time"], index_col="time")
    df_interpolated = df.interpolate(method="quadratic")

    nan_mask = df.isna()

    # Create plots
    fig, axs = plt.subplots(4, 2, figsize=(15, 6), sharex=True)

    # **Plot 1: Without interpolation (NaNs cause breaks in the line)**
    axs[0][0].plot(df.index, df["value_temp"], label="Temperature", color="red", linestyle="-", marker=".", alpha=0.7)
    axs[1][0].plot(df.index, df["value_hum"], label="Humidity", color="blue", linestyle="-", marker=".", alpha=0.7)
    axs[2][0].plot(df.index, df["value_acid"], label="Acidity", color="green", linestyle="-", marker=".", alpha=0.7)
    axs[3][0].plot(df.index, df["value_PV"], label="PV", color="orange", linestyle="-", marker=".", alpha=0.7)

    axs[0][0].set_title("Plot with Missing Data (NaNs cause breaks)")
    axs[0][0].legend()
    axs[0][0].grid(True)

    # **Plot 2: With Interpolation (Replacing NaNs with extrapolated values)**

    # axs[0][1].plot(df_interpolated.index, df_interpolated["value_temp"], label="Temperature", color="red", linestyle="-",
    #             marker=".", alpha=0.7)
    axs[1][1].plot(df_interpolated.index, df_interpolated["value_hum"], label="Humidity", color="blue", linestyle="-",
                marker=".", alpha=0.7)
    axs[2][1].plot(df_interpolated.index, df_interpolated["value_acid"], label="Acidity", color="green", linestyle="-",
                marker=".", alpha=0.7)
    axs[3][1].plot(df_interpolated.index, df_interpolated["value_PV"], label="PV", color="orange", linestyle="-",
                marker=".", alpha=0.7)

    axs[0][1].plot(df_interpolated.index[~nan_mask["value_temp"]], df_interpolated.loc[~nan_mask["value_temp"],"value_temp"], color="red", linestyle="None", marker="o", label="Interpolated Values")
    axs[0][1].plot(df_interpolated.index[nan_mask["value_temp"]], df_interpolated["value_temp"][nan_mask["value_temp"]], color=lighter_colors[0], linestyle="None", marker=".", label="Interpolated Values")


    for i in range(4):

        # cyberpunk(axs[i][0])
        # axs[i][0].grid(True)


        cyberpunk(axs[i][1])
        axs[i][1].grid(True)


    axs[0][1].set_title("Plot with Interpolated Data (NaNs replaced)")
    # axs[0][1].legend()


    # Formatting
    plt.xlabel("Timestamp")
    plt.xticks(rotation=30)
    plt.tight_layout()

    plt.show()


# def fun7():
#     file_name = "df_RuralIoT_23.csv"
#     path = Path("dane/original_files") / file_name
#     df = pd.read_csv(path)
#
#     # df["value_hum"] = df["value_hum"] - 75
#
#     out_path = Path("dane/humidity_ofset_fixed") / f"{path.name}_hum{path.suffix}"
#     df.to_csv(out_path)

def fun8(data_path: Path, save_path: Path = None):
    if save_path is None:
        save_path = data_path

    features = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

    graphic_path = save_path / "graphical_representation"
    graphic_path.mkdir(parents=True, exist_ok=True)

    paths = data_path.glob("*.csv")

    for path in paths:
        print(f"processing {path}")
        df = pd.read_csv(path)
        fig, axs = create_plot_many(rows=4, columns=1, subtitle=path.name, df=df, features=features)
        print(f"save to " + str(graphic_path / f'{path.name.split(".")[0]}.png'))
        plt.savefig(graphic_path / f'{path.name.split(".")[0]}.png', dpi=300, bbox_inches="tight")




def interpolate_segments():
    root_path = Path("dane/segment_representation_4h")
    save_path = Path("dane/segment_representation_4h_interpolated")
    columns_to_interpolate = ['value_temp', 'value_hum', 'value_acid', 'value_PV']

    for sensor_dir in root_path.iterdir():
        segments = sensor_dir.glob("*.csv")
        for segment in segments:
            print(segment.name)

            df = pd.read_csv(segment, parse_dates=["time"])
            df = df.sort_values("time")  # konieczne dla interpolacji czasowej
            df.set_index("time", inplace=True)

            # Interpolacja tylko wybranych kolumn (na podstawie czasu)
            df[columns_to_interpolate] = df[columns_to_interpolate].interpolate(method='time')

            df.reset_index(inplace=True)

            # 🔄 Tworzymy odpowiadającą ścieżkę zapisu
            out_dir = save_path / sensor_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            out_path = out_dir / segment.name
            df.to_csv(out_path, index=False)

            print(f"Zapisano do: {out_path}")











if __name__ == "__main__":
    # data_path8 = Path("dane") / "humidity_ofset_fixed" # path to files that will be saved as PNG
    # save_path8 = Path("dane") / "humidity_ofset_fixed" # path to files that will be saved as PNG
    # fun8(save_path8)


    # large_gap = 4*60*60
    # data_path = Path("dane") / "humidity_ofset_fixed"
    # save_path_segments = Path("dane") / "segment_representation_4h"
    # segment_creator(data_path=data_path, save_path=save_path_segments, large_gap_threshold=large_gap)

    # data_path8 = Path("dane") / "segment_representation_4h" / "df_RuralIoT_23" # path to files that will be saved as PNG
    # save_path8 = Path("dane") / "segment_representation_4h" / "df_RuralIoT_23" # path to files that will be saved as PNG
    # fun8(save_path8)

    # interpolate_segments()



    pass

