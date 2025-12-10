from oxford_ic_soh_preprocess import generate_ic_predictions_by_window

generate_ic_predictions_by_window(
    all_df=all_df,
    curve_cols=curve_cols,
    v_ref=v_ref,
    window_start_indices=window_start_indices,
    input_points_len=input_points_len,
    model=model,
    scaler_x=scaler_x,
    scaler_y=scaler_y,
    base_output_dir="predictions_by_window",
    cells_to_process=range(1, 9),
    n_points=600,
)
