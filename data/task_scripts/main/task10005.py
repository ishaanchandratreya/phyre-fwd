# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import phyre.creator as creator_lib

@creator_lib.define_task_template(
    left_table_scale=np.linspace(0.2, 0.3, 5),
    pot_scale=np.linspace(0.2, 0.3, 5),
    obstacle_scale=np.linspace(0.05, 0.1, 5),
    ball_scale=np.linspace(0.05, 0.1, 5),
)

def build_task(C, pot_scale, left_table_scale, obstacle_scale, ball_scale):

    eps_h = 0.01 * C.scene.height
    eps_w = 0.01 * C.scene.width


    table_left = C.add(
        'static bar',
        scale=left_table_scale,
        left=0.0
    )
    pot_guard_l = C.add(
        'static bar',
        scale=pot_scale,
        bottom=0,
        left=table_left.right,
        angle=90.)

    table_left.set_bottom(pot_guard_l.top)

    pot = C.add(
        'static jar',
        scale=pot_scale,
        left=pot_guard_l.right,
        bottom=0)

    pot_guard_r = C.add(
        'static bar',
        scale=pot_scale,
        bottom=0,
        left=pot.right,
        angle=90.)

    table = C.add(
        'static bar',
        scale=0.8,
        left=pot_guard_r.right,
        bottom=pot_guard_r.top,
    )

    obs_ball = C.add(
        'static ball',
        scale=obstacle_scale,
        left=table.left + 0.2 * C.scene.width,
        bottom=table.top,
    )

    if table.right < C.scene.width:
        ball_right = table.right

    else:
        ball_right = C.scene.width

    dyn_ball = C.add(
        'dynamic ball',
        scale=ball_scale,
        right=ball_right - 0.2 * C.scene.width,
        bottom=table.top,
    ).set_vel_x(-0.4 * C.scene.width).set_vel_y(0.3 * C.scene.height)

    C.update_task(
        body1=dyn_ball,
        body2=pot,
        relationships=[C.SpatialRelationship.TOUCHING],
    )

    C.set_meta(C.SolutionTier.BALL)
