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
    target_left=np.linspace(0.4, 0.7, 5),
    support_height=np.linspace(0.7, 0.8, 5),
    jar_scale=np.linspace(0.3, 0.4, 5),
    ball_scale=np.linspace(0.05, 0.1, 5),
)

def build_task(C, target_left, support_height, jar_scale, ball_scale):

    eps_h = 0.01 * C.scene.height
    eps_w = 0.01 * C.scene.width

    load = C.add('static bar', scale=0.3).set_bottom(support_height*C.scene.height)
    floor = C.add('static bar', scale=1).set_bottom(0)

    jar_guard_l = C.add(
        'static bar',
        scale=jar_scale,
        bottom=floor.top,
        left=target_left*C.scene.width,
        angle=90.)

    jar = C.add(
        'static jar',
        scale=jar_scale,
        left=jar_guard_l.right,
        bottom=floor.top)


    if jar.top > load.bottom:
        raise creator_lib.SkipTemplateParams

    if jar_guard_l.left < load.right:
        raise creator_lib.SkipTemplateParams

    jar_guard_r = C.add(
        'static bar',
        scale=jar_scale,
        bottom=floor.top,
        left=jar.right,
        angle=90.)

    ball = C.add('dynamic ball', scale=ball_scale).set_bottom(load.top + 0.1*eps_h).set_left(0)

    C.update_task(
        body1=ball,
        body2=jar,
        relationships=[C.SpatialRelationship.TOUCHING],
    )

    C.set_meta(C.SolutionTier.BALL)