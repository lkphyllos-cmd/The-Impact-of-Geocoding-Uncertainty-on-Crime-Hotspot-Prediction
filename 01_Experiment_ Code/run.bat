@echo off

SET "PYTHON_EXE=D:\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\python.exe"
SET "ROOT_PROJECT_DIR=E:\Crime_Uncertainty_Project\TGIS\Data"
SET "SOURCE_DATA_DIR=%ROOT_PROJECT_DIR%\00_Source_Data"
SET "EXPERIMENT_GROUPS_DIR=%ROOT_PROJECT_DIR%\02_Experiment_Groups"
SET "STUDY_AREA_SHP=%SOURCE_DATA_DIR%\study_area.shp"
SET "STREETS_SHP=%SOURCE_DATA_DIR%\Roadway_Functional__Dissolve_RouteName.shp"
SET "TRACT_SHP=%SOURCE_DATA_DIR%\Census_Tracts_in_2020.shp"
SET "NEIGHBORHOOD_SHP=%SOURCE_DATA_DIR%\Neighborhood_Clusters.shp"
SET "GRID_100_SHP=%SOURCE_DATA_DIR%\Grid_100m.shp"
SET "GRID_300_SHP=%SOURCE_DATA_DIR%\Grid_300m.shp"
SET "GRID_500_SHP=%SOURCE_DATA_DIR%\Grid_500m.shp"
SET "CELL_SIZE=50"
SET "BANDWIDTH=300"
SET "POINT_STREET_COL=BLOCK"
SET "STREET_LAYER_COL=ROUTENAME"

SET "TARGET_GROUP_DIR=%EXPERIMENT_GROUPS_DIR%\Short_Term\Group_4"


IF NOT EXIST "%TARGET_GROUP_DIR%" (echo [ERROR]  PAUSE & goto :eof)

"%PYTHON_EXE%" Experiment_for_one_Group.py ^
  --group_dir "%TARGET_GROUP_DIR%" ^
  --study_area "%STUDY_AREA_SHP%" ^
  --streets_shp "%STREETS_SHP%" ^
  --tract_shp "%TRACT_SHP%" ^
  --neighborhood_shp "%NEIGHBORHOOD_SHP%" ^
  --grid_100_shp "%GRID_100_SHP%" ^
  --grid_300_shp "%GRID_300_SHP%" ^
  --grid_500_shp "%GRID_500_SHP%" ^
  --point_street_col "%POINT_STREET_COL%" ^
  --street_layer_col "%STREET_LAYER_COL%" ^
  --cell_size %CELL_SIZE% ^
  --bandwidth %BANDWIDTH%


IF %ERRORLEVEL% NEQ 0 (ECHO [ERROR]! & PAUSE) ELSE (ECHO [SUCCESS]!)
echo.
PAUSE