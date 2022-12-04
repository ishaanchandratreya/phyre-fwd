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
    bar_height=np.linspace(0.2, 0.5, 5),
    bar_x=np.linspace(0.2, 0.8, 5),
    ball_size=np.linspace(0.05, 0.25, 5),
    ramp_size=np.linspace(0.05, 0.2, 5),
    search_params=dict(
        required_flags=['BALL:GOOD_STABLE'],
        excluded_flags=['BALL:TRIVIAL'],
        diversify_tier='ball',
    ),
    version='2',
)

@creator_lib.define_task_template(
    ball_scale=(0.05, 0.1, 5),
    ramp_angle=(45, 65, 5),
    ramp_size=(0.4, 0.5, 5),
)

def build_task(C, ball_scale, ramp_angle, ramp_size):

    eps_h = 0.01 * C.scene.height
    eps_w = 0.01 * C.scene.width

    target = C.add('static bar', scale=0.2).set_bottom(0)
    floor  = C.add('static bar', scale=0.8).set_bottom(0)

    target.set_left(0)
    floor.set_left(target.right)
    ramp = C.add('static bar', angle=ramp_angle, scale=ramp_size, right=C.scene.width, bottom=floor.top)

    ball = C.add('dynamic ball', scale=ball_scale).set_bottom(ramp.top + eps_h)

    C.update_task(
        body1=ball,
        body2=target,
        relationships=[C.SpatialRelationship.TOUCHING],
    )
    C.set_meta(C.SolutionTier.BALL)