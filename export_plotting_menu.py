# ===== Own Modules =====
import functions as fn
from settings import setting
from export_plotting import (
    TensorBoardExporter,
    UMAPPlotter,
    ConfusionMatrixPlotter,
    TensorBoardPlotter
)


class ExportPlottingMenu:

    # Export & Plotting Menu - Tools for exporting and visualizing results.

    #############################################################################################################
    # CONSTRUCTOR

    def __init__(self) -> None:
        pass

    #############################################################################################################
    # METHODS

    # Display the Export & Plotting submenu and route to the selected option.
    def menu(self) -> None:
        while True:
            print("\n:EXPORT & PLOTTING MENU:")
            print("1) Export Training Metrics to Excel (TensorBoard → Excel)")
            print("2) Plot UMAP/t-SNE/PaCMAP (Publication-ready)")
            print("3) Plot Confusion Matrix (Publication-ready)")
            print("4) Plot Training Curves (TensorBoard → Publication-ready)")
            print("5) Back to Main Menu")

            choice = fn.input_int("Please choose: ")

            if choice == 1:
                self._run_excel_exporter()
            elif choice == 2:
                self._run_umap_plotter()
            elif choice == 3:
                self._run_confusion_matrix_plotter()
            elif choice == 4:
                self._run_training_metrics_plotter()
            elif choice == 5:
                print("\nReturning to main menu...")
                break
            else:
                print("Not a valid option!")

    # Run the Excel Exporter.
    def _run_excel_exporter(self) -> None:
        print("\n:EXPORT TRAINING METRICS TO EXCEL:")
        
        # Get mode from settings
        mode = setting.get('export_mode', 'auto')
        
        if mode == 'crossval':
            print("  Mode: CROSS-VALIDATION")
            print("  Enter the path to the cross_validation/ folder")
            print("  (contains dataset_01/, dataset_02/, ...)")
            print("  Example: output/cross_validation/")
        elif mode == 'single':
            print("  Mode: SINGLE TRAINING")
            print("  Enter the path to the logs/ folder")
            print("  Example: output/train/20260909-143022/logs/")
        else:
            print("  Mode: AUTO (will detect from folder structure)")
            print("  Enter the path to the logs folder (single training)")
            print("  or the cross_validation/ folder (cross-validation)")
        
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            if mode == 'crossval':
                prompt = "Enter path to cross_validation folder: "
            elif mode == 'single':
                prompt = "Enter path to logs folder: "
            else:
                prompt = "Enter path to TensorBoard logs folder: "
            
            logdir = input(prompt).strip()
            if not logdir:
                print("Cancelled.")
                return

            exporter = TensorBoardExporter(
                logdir=logdir,
                output_file=setting['pth_output'] / "train_metrics.xlsx",
                prob_dir=None,
                roc_epoch=setting.get('export_excel_roc_epoch', 'balanced_accuracy'),
                pr_epoch=setting.get('export_excel_pr_epoch', 'balanced_accuracy'),
                mode=mode
            )
            exporter.export_with_charts()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the UMAP Plotter.
    def _run_umap_plotter(self) -> None:
        print("\n:PLOT UMAP/t-SNE/PaCMAP:")
        print("  Input: input/ (CSV files with embedding coordinates)")
        print("  Output: output/ (publication-ready plots)")
        print("  Note: Supports UMAP, t-SNE, TriMAP, and PaCMAP.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            plotter = UMAPPlotter(
                input_folder=setting['pth_input'],
                output_folder=setting['pth_output'],
                output_format=setting.get('export_umap_format', 'tiff'),
                raster_dpi=setting.get('export_umap_dpi', 600),
                point_size=setting.get('export_umap_point_size', 30),
                point_alpha=setting.get('export_umap_point_alpha', 0.5),
                show_ellipses=setting.get('export_umap_show_ellipses', False),
                palette=setting.get('export_umap_palette', 'jet'),
                fixed_aspect_ratio=setting.get('export_umap_fixed_aspect', True),
                axes_height_inches=setting.get('export_umap_axes_height', 5.0),
                show_legend=setting.get('export_umap_show_legend', True),
                legend_position=setting.get('export_umap_legend_position', 'outside'),
                legend_marker_size=setting.get('export_umap_legend_marker_size', 100),
                font_family=setting.get('export_umap_font_family', 'Arial'),
                axis_label_size=setting.get('export_umap_axis_label_size', 22),
                legend_font_size=setting.get('export_umap_legend_font_size', 22),
                tick_label_size=setting.get('export_umap_tick_label_size', 22),
                show_grid=setting.get('export_umap_show_grid', True),
                grid_alpha=setting.get('export_umap_grid_alpha', 0.3),
                left_margin=setting.get('export_umap_left_margin', 1.2),
                right_margin=setting.get('export_umap_right_margin', 1.0),
                bottom_margin=setting.get('export_umap_bottom_margin', 0.9),
                top_margin=setting.get('export_umap_top_margin', 0.5),
                legend_margin_extra=setting.get('export_umap_legend_margin_extra', 3.0)
            )
            plotter.process_all_files()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()


    # Run the Confusion Matrix Plotter.
    def _run_confusion_matrix_plotter(self) -> None:
        print("\n:PLOT CONFUSION MATRIX:")
        print("  Input: input/ (JSON files with confusion matrix data)")
        print("  Output: output/ (publication-ready plots)")
        print("  Note: Supports raw and normalized confusion matrices.")
        print()

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            plotter = ConfusionMatrixPlotter(
                input_folder=setting['pth_input'],
                output_folder=setting['pth_output'],
                output_format=setting.get('export_cm_format', 'tiff'),
                raster_dpi=setting.get('export_cm_dpi', 600),
                normalize=setting.get('export_cm_normalize', 'rows'),
                show_counts=setting.get('export_cm_show_counts', True),
                combined_mode=setting.get('export_cm_combined', False),
                use_fixed_axes_height=setting.get('export_cm_use_fixed_height', True),
                fixed_axes_height=setting.get('export_cm_fixed_height', 6.0),
                cmap=setting.get('export_cm_cmap', 'Blues'),
                show_title=setting.get('export_cm_show_title', True),
                show_axis_labels=setting.get('export_cm_show_axis_labels', True),
                show_colorbar=setting.get('export_cm_show_colorbar', True),
                show_per_class_accuracy=setting.get('export_cm_show_per_class_acc', False),
                show_overall_accuracy=setting.get('export_cm_show_overall_acc', True),
                font_family=setting.get('export_cm_font_family', 'Arial'),
                master_font_size=setting.get('export_cm_master_font_size', None),
                axis_label_size=setting.get('export_cm_axis_label_size', 30),
                title_font_size=setting.get('export_cm_title_font_size', 30),
                tick_label_size=setting.get('export_cm_tick_label_size', 25),
                annotation_font_size=setting.get('export_cm_annotation_font_size', 16),
                legend_font_size=setting.get('export_cm_legend_font_size', 30),
                xtick_rotation=setting.get('export_cm_xtick_rotation', 45),
                ytick_rotation=setting.get('export_cm_ytick_rotation', 0),
                annotation_decimal_places=setting.get('export_cm_annotation_decimal_places', 3)
            )
            plotter.process_all_files()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

    # Run the Training Metrics Plotter.
    def _run_training_metrics_plotter(self) -> None:
        print("\n:PLOT TRAINING CURVES:")
        
        # Get mode from settings
        mode = setting.get('export_mode', 'auto')
        
        if mode == 'crossval':
            print("  Mode: CROSS-VALIDATION")
            print("  Enter the path to the cross_validation/ folder")
            print("  (contains dataset_01/, dataset_02/, ...)")
            print("  Example: output/cross_validation/")
        elif mode == 'single':
            print("  Mode: SINGLE TRAINING")
            print("  Enter the path to the logs/ folder")
            print("  Example: output/train/20260909-143022/logs/")
        else:
            print("  Mode: AUTO (will detect from folder structure)")
            print("  Enter the path to the logs folder (single training)")
            print("  or the cross_validation/ folder (cross-validation)")
        
        print()

        if mode == 'crossval':
            prompt = "Enter path to cross_validation folder: "
        elif mode == 'single':
            prompt = "Enter path to logs folder: "
        else:
            prompt = "Enter path to TensorBoard logs folder: "
        
        logdir = input(prompt).strip()
        if not logdir:
            print("Cancelled.")
            return

        confirm = input("Continue? (yes/no): ").strip().lower()
        if confirm not in ['yes', 'y']:
            print("Operation cancelled.")
            return

        try:
            plotter = TensorBoardPlotter(
                logdir=logdir,
                output_folder=setting['pth_output'],
                output_format=setting.get('export_train_format', 'tiff'),
                raster_dpi=setting.get('export_train_dpi', 300),
                mode=setting.get('export_mode', 'auto'),
                roc_curve_epoch=setting.get('export_train_roc_epoch', 'balanced_accuracy'),
                pr_curve_epoch=setting.get('export_train_pr_epoch', 'balanced_accuracy'),
                use_fixed_axes_height=setting.get('export_train_use_fixed_height', True),
                fixed_axes_height=setting.get('export_train_fixed_height', 5.0),
                master_font_size=setting.get('export_train_master_font_size', 22),
                line_width=setting.get('export_train_line_width', 2.0),
                show_markers=setting.get('export_train_show_markers', True),
                marker_size=setting.get('export_train_marker_size', 5),
                marker_frequency=setting.get('export_train_marker_frequency', 1),
                smoothing=setting.get('export_train_smoothing', 0.0),
                font_family=setting.get('export_train_font_family', 'Arial'),
                axis_label_size=setting.get('export_train_axis_label_size', 12),
                title_font_size=setting.get('export_train_title_font_size', 14),
                tick_label_size=setting.get('export_train_tick_label_size', 10),
                legend_font_size=setting.get('export_train_legend_font_size', 10),
                show_legend=setting.get('export_train_show_legend', True),
                show_grid=setting.get('export_train_show_grid', True),
                grid_alpha=setting.get('export_train_grid_alpha', 0.3),
                grid_linestyle=setting.get('export_train_grid_linestyle', '--'),
                grid_color=setting.get('export_train_grid_color', 'gray'),
                min_class_acc_threshold=setting.get('export_train_min_class_acc_threshold', 0.65),
                # Plot selection
                plot_loss=setting.get('export_train_plot_loss', True),
                plot_accuracy=setting.get('export_train_plot_accuracy', True),
                plot_f1=setting.get('export_train_plot_f1', True),
                plot_lr=setting.get('export_train_plot_lr', False),
                plot_auc=setting.get('export_train_plot_auc', False),
                plot_ap=setting.get('export_train_plot_ap', False),
                plot_per_class_accuracy=setting.get('export_train_plot_per_class_acc', True),
                plot_class_weights=setting.get('export_train_plot_class_weights', False),
                plot_class_counts=setting.get('export_train_plot_class_counts', False),
                plot_composite_score=setting.get('export_train_plot_composite', False),
                plot_balanced_accuracy=setting.get('export_train_plot_balanced_acc', True),
                plot_class_std=setting.get('export_train_plot_class_std', False),
                plot_min_class_acc=setting.get('export_train_plot_min_class_acc', False),
                plot_gpu_memory=setting.get('export_train_plot_gpu_memory', False),
                plot_roc_curves=setting.get('export_train_plot_roc', True),
                plot_pr_curves=setting.get('export_train_plot_pr', True),
                # Legend outside settings
                per_class_legend_outside=setting.get('export_train_per_class_legend_outside', True),
                roc_legend_outside=setting.get('export_train_roc_legend_outside', True),
                pr_legend_outside=setting.get('export_train_pr_legend_outside', True),
                # Legend positions
                loss_legend_loc=setting.get('export_train_loss_legend_loc', 'upper right'),
                accuracy_legend_loc=setting.get('export_train_accuracy_legend_loc', 'lower right'),
                f1_legend_loc=setting.get('export_train_f1_legend_loc', 'lower right'),
                per_class_legend_loc=setting.get('export_train_per_class_legend_loc', 'best'),
                roc_legend_loc=setting.get('export_train_roc_legend_loc', 'lower right'),
                pr_legend_loc=setting.get('export_train_pr_legend_loc', 'lower left'),
                class_std_legend_loc=setting.get('export_train_class_std_legend_loc', 'upper right'),
                min_class_acc_legend_loc=setting.get('export_train_min_class_acc_legend_loc', 'lower right'),
                composite_legend_loc=setting.get('export_train_composite_legend_loc', 'lower right'),
                balanced_acc_legend_loc=setting.get('export_train_balanced_acc_legend_loc', 'lower right'),
                gpu_memory_legend_loc=setting.get('export_train_gpu_memory_legend_loc', 'best'),
                lr_legend_loc=setting.get('export_train_lr_legend_loc', 'best'),
                auc_legend_loc=setting.get('export_train_auc_legend_loc', 'lower right'),
                ap_legend_loc=setting.get('export_train_ap_legend_loc', 'lower right')
            )

            if plotter.process_runs():
                plotter.generate_all_plots()
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()