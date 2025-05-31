from main_config import COLORS_TERMINAL , HYPERPARAMETER_SWEEP, COLORS_BOLD, type_out , print_cursor , show_intro , show_outro

from chap_4_sculpting_the_Ideal import part_1_fine_tuning_best_models
from chap_4_sculpting_the_Ideal import part_2_results_fine_tuning
from chap_4_sculpting_the_Ideal import part_3_loading_models
from chap_5_awakening_the_neural_mind import part_1_building_the_mind
from chap_5_awakening_the_neural_mind import part_2_tensorflow
from chap_6_cracking_the_black_box import part_1_shap

import sys
import traceback

CHAP = 3

if __name__ == "__main__":
    print("\033[2J\033[H", end='')
    show_intro()
    PLAYING = True
    while PLAYING:
        try:
            type_out("""
                \nAvailable chapters:
                \n-------------------
                \n0: The Genesis of Data
                \n1: Peering into the Data Abyss
                \n2: Cleansing the Noise
                \n3: The Hunt for the Right Model
                \n4: Sculpting the Ideal Model
                \n5: Awakening the Neural Mind
                \n6: Cracking the black box
                \n-------------------
                \n9: Load model from a file
                \n10: Exit the project
            """,delay=0.01)
            type_out("\nEnter the chapter number you want to run: ")
            CHAP = int(print_cursor())
        except ValueError:
            print(f"{COLORS_TERMINAL['RED']}Invalid input. Please enter a valid chapter number (0, 1, 2, 3, or 4).{COLORS_TERMINAL['RESET_COLOR']}")
            PLAYING = False
            sys.exit(1)

        try:
            # --------------------------------------------------------------------------------------- #
            if CHAP == 0:
                print("Running chapter 0: The Genesis of Data")
                from chap_0_the_genesis_of_data import part_1_test_datasets
            
            # --------------------------------------------------------------------------------------- #
            elif CHAP == 1:
                print("Running chapter 1: Peering into the Data Abyss")
                from chap_1_peering_into_the_data_abyss import part_1_exploration
                from chap_1_peering_into_the_data_abyss import part_2_exploration
                from chap_1_peering_into_the_data_abyss import part_3_preprocesssing
                from chap_1_peering_into_the_data_abyss import part_4_visualisations
                from chap_1_peering_into_the_data_abyss import part_5_visualisations
                from chap_1_peering_into_the_data_abyss import part_6_visualisations
                from chap_1_peering_into_the_data_abyss import part_7_exploration
            
            # --------------------------------------------------------------------------------------- #
            elif CHAP == 2:
                print("Running chapter 2: Cleansing the Noise")
                from chap_2_cleansing_the_noise import part_1_cleaning
                from chap_2_cleansing_the_noise import part_2_visualisations
                from chap_2_cleansing_the_noise import part_3_cleaning

            # --------------------------------------------------------------------------------------- #
            elif CHAP == 3:
                print("Running chapter 3: The Hunt for the Right Model")
                from chap_3_the_hunt_for_the_right_model import part_1_regression
                from chap_3_the_hunt_for_the_right_model import part_2_democracy_ensemble_methods
                from chap_3_the_hunt_for_the_right_model import part_3_results_regression
                from chap_3_the_hunt_for_the_right_model import part_4_results_ensemble_methods
            
            # --------------------------------------------------------------------------------------- #
            elif CHAP == 4:
                type_out("""
                    \nWhich sweep of hyperparameters do you want to run?
                    \n---------------------------------------------------
                    \n1: Sweep 1 - Hyperparameter Sweep - Random Search
                    \n2: Sweep 2 - Hyperparameter Sweep - Grid Search
                    \n3: Sweep 3 - Hyperparameter Sweep - ensemble methods
                    \n---------------------------------------------------
                    \n10: Back to main menu
                    """)
                try:
                    HYPERPARAMETER_SWEEP = int(print_cursor())
                    if HYPERPARAMETER_SWEEP in [1, 2, 3]:
                        print(f"Running chapter 4: Sculpting the Ideal Model - Sweep {HYPERPARAMETER_SWEEP}")
                        pipeline, params, search_method, model_prefix, best_models = part_1_fine_tuning_best_models.enter_hyperparameter_sweep(HYPERPARAMETER_SWEEP)
                        results, new_best_models = part_1_fine_tuning_best_models.main(pipeline, params, search_method, model_prefix, best_models)
                        part_1_fine_tuning_best_models.plot_results(results, new_best_models,model_prefix)
                        part_2_results_fine_tuning.main(model_prefix)

                    elif HYPERPARAMETER_SWEEP == 10:
                        print(f"{COLORS_BOLD['BOLD_GREEN']}Returning to main menu...{COLORS_TERMINAL['RESET_COLOR']}")
                        continue

                    else:
                        raise ValueError

                except ValueError:
                    print(f"{COLORS_TERMINAL['RED']}Invalid input. Please enter 1 , 2 , 3 or 10.{COLORS_TERMINAL['RESET_COLOR']}")
                    PLAYING = False
                    sys.exit(1)


            # --------------------------------------------------------------------------------------- #
            elif CHAP == 5:
                print("Running chapter 5: Awakening the neural mind")

                type_out("""
                    \nWhich part of the chapter do you want to run?
                    \n---------------------------------------------------
                    \n1: Part 1 - Building the Neural Mind
                    \n2: Part 2 - TensorFlow Neural Network
                    \n3: Part 3 - Loading and Testing Models
                    \n---------------------------------------------------
                    \n10: Back to main menu
                """)
                user_choice = int(print_cursor())
                if user_choice == 1:
                    print(f"{COLORS_BOLD['BOLD_GREEN']}Running Part 1: Building the Neural Mind{COLORS_TERMINAL['RESET_COLOR']}")
                    results, models, model_prefix = part_1_building_the_mind.main()
                    part_1_building_the_mind.plot_results(results, models, model_prefix)

                elif user_choice == 2:
                    print(f"{COLORS_BOLD['BOLD_GREEN']}Running Part 2: TensorFlow Neural Network{COLORS_TERMINAL['RESET_COLOR']}")
                    part_2_tensorflow.main()

                elif user_choice == 3:
                    print(f"{COLORS_BOLD['BOLD_GREEN']}Running Part 3: Loading and Testing Models{COLORS_TERMINAL['RESET_COLOR']}")
                    models, model_prefix = part_3_loading_models.main()
                    results = part_3_loading_models.test_loaded_models(models)
                    if not results:
                        print(f"{COLORS_TERMINAL['RED']}Error testing loaded models.{COLORS_TERMINAL['RESET_COLOR']}")
                        PLAYING = False
                        sys.exit(1)
                    else:
                        print(f"{COLORS_BOLD['BOLD_GREEN']}Models loaded and tested successfully!{COLORS_TERMINAL['RESET_COLOR']}")
                        part_1_fine_tuning_best_models.plot_results(results, models, model_prefix, save_models=False)

            elif CHAP == 6:
                print("Running chapter 6: Cracking the black box")
                part_1_shap.main()

            # --------------------------------------------------------------------------------------- #
            elif CHAP == 9:
                models , model_prefix = part_3_loading_models.main()
                results = part_3_loading_models.test_loaded_models(models)
                if not results:
                    print(f"{COLORS_TERMINAL['RED']}Error testing loaded models.{COLORS_TERMINAL['RESET_COLOR']}")
                    PLAYING = False
                    sys.exit(1)
                else:
                    print(f"{COLORS_BOLD['BOLD_GREEN']}Models loaded and tested successfully!{COLORS_TERMINAL['RESET_COLOR']}")
                    part_1_fine_tuning_best_models.plot_results(results,models,model_prefix,save_models=False)
                    part_2_results_fine_tuning.main(model_prefix)


            # --------------------------------------------------------------------------------------- #
            elif CHAP == 10:
                print(f"{COLORS_BOLD['BOLD_GREEN']}Exiting the project. Thank you for participating!{COLORS_TERMINAL['RESET_COLOR']}")
                PLAYING = False
                break

            else:
                print(f"{COLORS_TERMINAL['RED']}Invalid chapter number. Please enter a valid chapter number (0, 1, 2, 3, or 4).{COLORS_TERMINAL['RESET_COLOR']}")
                PLAYING = False
                sys.exit(1)

        except Exception as e:
            print(f"{COLORS_TERMINAL['RED']}An error occurred: {e}{COLORS_TERMINAL['RESET_COLOR']}")
            print(f"\n\n {traceback.format_exc()} \n\n")
            PLAYING = False
            sys.exit(1)
        except KeyboardInterrupt:
            print(f"\n{COLORS_TERMINAL['RED']}KeyboardInterrupt detected. Exiting the project.{COLORS_TERMINAL['RESET_COLOR']}")
            PLAYING = False
            sys.exit(1)
        
    show_outro()



