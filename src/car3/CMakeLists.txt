cmake_minimum_required(VERSION 2.8.3)

project(car3)

find_package(catkin REQUIRED)

catkin_package()

find_package(roslaunch)

catkin_install_python(PROGRAMS
  scripts/gripper_mimic.py
  scripts/grasp_attach.py
  scripts/spawn_cubes.py
  scripts/cube_vision.py
  scripts/calibrate_vision_ground_truth.py
  scripts/test_observe_grasp.py
  scripts/pick_place_executor.py
  scripts/test_move_base_goal.py
  scripts/mark_cube_areas.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

foreach(dir config launch meshes urdf models world)
	install(DIRECTORY ${dir}/
		DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/${dir})
endforeach(dir)


if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_cube_vision_offline.py)
endif()
