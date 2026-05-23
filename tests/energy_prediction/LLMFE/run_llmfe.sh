################ LLM-FE with API ################
export API_KEY=ENTER API KEY
#### Classification Datasets ####

## balance-scale ## 
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name balance-scale --spec_path ./specs/specification_balance-scale.txt --log_path ./logs/balance-scale_gpt3.5

## breast-w ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name breast-w --spec_path ./specs/specification_breast-w.txt --log_path ./logs/breast-w_gpt3.5

## blood ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name blood --spec_path ./specs/specification_blood.txt --log_path ./logs/blood_gpt3.5

## car ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name car --spec_path ./specs/specification_car.txt --log_path ./logs/car_gpt3.5

## cmc ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name cmc --spec_path ./specs/specification_cmc.txt --log_path ./logs/cmc_gpt3.5

## credit-g ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name credit-g --spec_path ./specs/specification_credit-g.txt --log_path ./logs/credit-g_gpt3.5

## eucalyptus ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name eucalyptus --spec_path ./specs/specification_eucalyptus.txt --log_path ./logs/eucalyptus_gpt3.5

## heart ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name heart --spec_path ./specs/specification_heart.txt --log_path ./logs/heart_gpt3.5

## pc1 ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name pc1 --spec_path ./specs/specification_pc1.txt --log_path ./logs/pc1_gpt3.5

## tic-tac-toe ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name tic-tac-toe --spec_path ./specs/specification_tic-tac-toe.txt --log_path ./logs/tic-tac-toe_gpt3.5

## vehicle ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name vehicle --spec_path ./specs/specification_vehicle.txt --log_path ./logs/vehicle_gpt3.5

#### Large-Scale Classification Datasets ####

## adult ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name junglechess --spec_path ./specs/specification_junglechess.txt --log_path ./logs/junglechess_gpt3.5

## bank ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name adult --spec_path ./specs/specification_adult.txt --log_path ./logs/adult_gpt3.5

## junglechess ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name bank --spec_path ./specs/specification_bank.txt --log_path ./logs/bank_gpt3.5

## diabetes ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name diabetes --spec_path ./specs/specification_diabetes.txt --log_path ./logs/diabetes_gpt3.5

#### High-Dimensional Classification Datasets ####

## covtype ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name covtype --spec_path ./specs/specification_covtype.txt --log_path ./logs/covtype_gpt3.5

## myocardial ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name myocardial --spec_path ./specs/specification_myocardial.txt --log_path ./logs/myocardial_gpt3.5

## communities ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name communities --spec_path ./specs/specification_communities.txt --log_path ./logs/communities_gpt3.5

## arrhythmia ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name arrhythmia --spec_path ./specs/specification_arrhythmia.txt --log_path ./logs/arrhythmia_gpt3.5


#### Regression Datasets ####

## bike ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name bike --spec_path ./specs/specification_bike.txt --log_path ./logs/bike_gpt3.5

## crab ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name crab --spec_path ./specs/specification_crab.txt --log_path ./logs/crab_gpt3.5

## housing ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name housing --spec_path ./specs/specification_housing.txt --log_path ./logs/housing_gpt3.5

## insurance ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name insurance --spec_path ./specs/specification_insurance.txt --log_path ./logs/insurance_gpt3.5

## forest-fires ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name forest-fires --spec_path ./specs/specification_forest-fires.txt --log_path ./logs/forest-fires_gpt3.5_tpfn

## wine ##
# python main.py --use_api True --api_model "gpt-3.5-turbo" --problem_name wine --spec_path ./specs/specification_wine.txt --log_path ./logs/wine_gpt3.5

################ LLM-FE with LOCAL LLM ################

#### Classification Datasets ####

## balance-scale ##
# python main.py --problem_name balance-scale --spec_path ./specs/specification_balance-scale.txt --log_path balance-scale

## breast-w ##
# python main.py --problem_name breast-w --spec_path ./specs/specification_breast-w.txt --log_path breast-w

## blood ##
# python main.py --problem_name blood --spec_path ./specs/specification_blood.txt --log_path blood

## car ##
# python main.py --problem_name car --spec_path ./specs/specification_car.txt --log_path car

## cmc ##
# python main.py --problem_name cmc --spec_path ./specs/specification_cmc.txt --log_path cmc

## credit-g ##
# python main.py --problem_name credit-g --spec_path ./specs/specification_credit-g.txt --log_path credit-g

## eucalyptus ##
# python main.py --problem_name eucalyptus --spec_path ./specs/specification_eucalyptus.txt --log_path eucalyptus

## heart ##
# python main.py --problem_name heart --spec_path ./specs/specification_heart.txt --log_path heart

## pc1 ##
# python main.py --problem_name pc1 --spec_path ./specs/specification_pc1.txt --log_path pc1

## tic-tac-toe ##
# python main.py --problem_name tic-tac-toe --spec_path ./specs/specification_tic-tac-toe.txt --log_path tic-tac-toe

## vehicle ##
# python main.py --problem_name vehicle --spec_path ./specs/specification_vehicle.txt --log_path vehicle

#### Regression Datasets ####

## bike ##
# python main.py --problem_name bike --spec_path ./specs/specification_bike.txt --log_path bike

## crab ##
# python main.py --problem_name crab --spec_path ./specs/specification_crab.txt --log_path crab

## housing ##
# python main.py --problem_name housing --spec_path ./specs/specification_housing.txt --log_path housing

## insurance ##
# python main.py --problem_name insurance --spec_path ./specs/specification_insurance.txt --log_path insurance

## forest-fires ##
# python main.py --problem_name forest-fires --spec_path ./specs/specification_forest-fires.txt --log_path forest-fires

## wine ##
# python main.py --problem_name wine --spec_path ./specs/specification_wine.txt --log_path wine