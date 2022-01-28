// Copyright (c) Facebook, Inc. and its affiliates.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
#include "task_utils.h"
#include "task_validation.h"
#include "thrift_box2d_conversion.h"

#include <iostream>
#include <random>

namespace {
struct SimulationRequest {
  int maxSteps;
  int stride;
  int perturbStep = -1;  // Perturb the simulation at this point, -1 => don't
  bool stopAfterSolved = true;  // Stop the simulation once solved
  int perturbType = -1; // Type of perturbation involved (for stochasticity of environment)
};

void perturbSimulation(std::unique_ptr<b2WorldWithData> &world, const SimulationRequest &request) {
  // Takes the current world, and applies some random
  // perturbations to the simulation.
  // TODO(rgirdhar): Support other perturbations than just flipping the gravity
  // TODO(rgirdhar): Expose interface to SimulationRequest to specify kind of
  //   perturbation etc?


  //perturb_type 0 corresponds to gravity flip
  if (request.perturbType == 0){
    auto cur_grav = world->GetGravity();
    cur_grav = -cur_grav;
    world->SetGravity(cur_grav);
  }


  //perturb type 1 corresponds to fluctuations in gravity
  //TODO: function may be buggy (check sign on gravity)
  if (request.perturbType == 1){

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0.0, 0.1);
    auto cur_grav = world->GetGravity();

    if (cur_grav.y > 10){
        cur_grav += dis(gen) * cur_grav;
    }
    else if (cur_grav.y > 9){
        cur_grav -= dis(gen) * cur_grav;
    }
    else {
        cur_grav = world->GetGravity();
    }

    world->SetGravity(cur_grav);

  }
}

// Runs simulation for the scene. If task is not nullptr, is-task-solved checks
// are performed.
::task::TaskSimulation simulateTask(const ::scene::Scene &scene,
                                    const SimulationRequest &request,
                                    const ::task::Task *task) {
  std::unique_ptr<b2WorldWithData> world = convertSceneToBox2dWorld(scene);

  /*
  b2Body* c_body = world->GetBodyList();


  while (c_body != nullptr){

    std::cout << "type" << c_body->GetType() << std::endl;
    std::cout << "lin damping" << c_body->GetLinearDamping() << std::endl;
    std::cout << "ang damping" << c_body->GetAngularDamping() << std::endl;
    std::cout << "gravity scale" << c_body->GetGravityScale() << std::endl;
    std::cout << "mass" << c_body->GetMass() << std::endl;

    c_body = c_body -> GetNext();
  }
  */




  bool warmStart = true;
  world->SetWarmStarting(warmStart);

  //std::cout << "Does the world auto clear forces" << world->GetAutoClearForces() << std::endl;
  //std::cout << "Does the world substep" << world->GetSubStepping() << std::endl;
  //std::cout << "Does the world support continuous physics" << world->GetContinuousPhysics() << std::endl;

  unsigned int continuousSolvedCount = 0;
  std::vector<::scene::Scene> scenes;
  std::vector<bool> solveStateList;
  bool solved = false;
  int step = 0;

  // For different relations number of steps the condition should hold varies.
  // For NOT_TOUCHING relation one of three should be true:
  //   1. Objects are touching at the beginning and then not touching for
  //   kStepsForSolution steps.
  //   2. Objects are not touching at the beginning, touching at some point of
  //   simulation and then not touching for kStepsForSolution steps.
  //   3. Objects are not touching whole sumulation.
  // For TOUCHING_BRIEFLY a single touching is allowed.
  // For all other relations the condition must hold for kStepsForSolution
  // consequent steps.
  bool lookingForSolution =
      (task == nullptr || !isTaskInSolvedState(*task, *world) ||
       task->relationships.size() != 1 ||
       task->relationships[0] != ::task::SpatialRelationship::NOT_TOUCHING);
  const bool allowInstantSolution =
      (task != nullptr && task->relationships.size() == 1 &&
       task->relationships[0] == ::task::SpatialRelationship::TOUCHING_BRIEFLY);

  //b2ContactManager cm = world->GetContactManager();
  //b2ContactManager* other_cm = const_cast<b2ContactManager*>(&cm);
  //other_cm->FindNewContacts();

  scenes.push_back(updateSceneFromWorld(scene, *world));
  //std::cout << "Number of contacts at timestep " << step << "is" << world->GetContactCount() << std::endl;
  //std::cout << "Number of joint at timestep " << step << "is" << world->GetJointCount() << std::endl;
  //TODO: Do it at non zero time-step (if collision detected in previous)
  world->Step(kTimeStep/100, 1, 1);
  for (; step < request.maxSteps; step++) {
    // Instruct the world to perform a single step of simulation.
    // It is generally best to keep the time step and iterations fixed.
    world->Step(kTimeStep, kVelocityIterations, kPositionIterations);

    //std::cout << "Number of contacts at timestep " << step << "is" << world->GetContactCount() << std::endl;
    //std::cout << "Number of joints at timestep " << step << "is" << world->GetJointCount() << std::endl;
    if (step == request.perturbStep && request.perturbStep != -1) {
      perturbSimulation(world, request);
    }
    if (request.stride > 0 && (step+1) % request.stride == 0) {
      scenes.push_back(updateSceneFromWorld(scene, *world));
    }
    if (task == nullptr) {
      solveStateList.push_back(false);
    } else {
      solveStateList.push_back(isTaskInSolvedState(*task, *world));
      if (solveStateList.back()) {
        continuousSolvedCount++;
        if (lookingForSolution) {
          if (continuousSolvedCount >= kStepsForSolution ||
              allowInstantSolution) {
            solved = true;
            // Stop the simulation and return if solved. Added this flag to
            // optionally let the simulation run, since it might make learning
            // forward models a bit more stable.
            if (request.stopAfterSolved) {
              //if you are here this means that simulation is in "done" state
              break;
            }
          }
        }
      } else {
        lookingForSolution = true;  // Task passed through non-solved state.
        continuousSolvedCount = 0;
      }
    }
  }

  if (!lookingForSolution && continuousSolvedCount == solveStateList.size()) {
    // See condition 3) for NOT_TOUCHING relation above.
    solved = true;
  }

  {
    std::vector<bool> stridedSolveStateList;
    if (request.stride > 0) {
      for (size_t i = 0; i < solveStateList.size(); i += request.stride) {
        stridedSolveStateList.push_back(solveStateList[i]);
      }
    }
    stridedSolveStateList.swap(solveStateList);
  }

  ::task::TaskSimulation taskSimulation;
  taskSimulation.__set_sceneList(scenes);
  taskSimulation.__set_stepsSimulated(step);
  if (task != nullptr) {
    taskSimulation.__set_solvedStateList(solveStateList);
    taskSimulation.__set_isSolution(solved);
  }

  return taskSimulation;
}
}  // namespace

std::vector<::scene::Scene> simulateScene(const ::scene::Scene &scene,
                                          const int num_steps) {
  const SimulationRequest request{num_steps, 1};
  const auto simulation = simulateTask(scene, request, /*task=*/nullptr);
  return simulation.sceneList;
}

::task::TaskSimulation simulateTask(const ::task::Task &task,
                                    const int num_steps, const int stride,
                                    const int perturb_step,
                                    const bool stop_after_solved,
                                    const int perturb_type) {
  const SimulationRequest request{
    num_steps, stride, perturb_step, stop_after_solved, perturb_type};
  return simulateTask(task.scene, request, &task);
}