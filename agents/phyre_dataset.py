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
"""The PHYRE dataset loader for mp reads."""
import logging
import math
import numpy as np
import torch
import hydra
import scipy.spatial
import os
import phyre
import im_fwd_agent
#import obj_nets

# There is an open issue wrt the memory leak in the data loader
# https://github.com/pytorch/pytorch/issues/13246
# For now just sticking to using np arrays as much as possible throughout.

class PhyreVideoSavedDataset(torch.utils.data.Dataset):

    def __init__(self,
                 write_path,
                 tier,
                 task_ids,
                 task_indices,
                 is_solved,
                 actions,
                 simulator_cfg,
                 mode='train',
                 balance_classes=True,
                 hard_negatives=0.0,
                 init_clip_ratio_to_sim=-1,
                 init_frames_to_sim=0,
                 frames_per_clip=1,
                 n_hist_frames=3,
                 shuffle_indices=False,
                 drop_objs='',
                 obj_fwd_model=None):


        self.write_path = write_path
        self.tier = tier
        self.task_ids = np.array(task_ids)
        self.task_indices = np.array(task_indices)
        self.is_solved = np.array(is_solved)
        self.actions = (actions.cpu().numpy()
                        if torch.is_tensor(actions) else np.array(actions))
        self.simulator_cfg = simulator_cfg
        self.mode = mode
        if self.mode == 'train':
            logging.info('Data set: size=%d, positive_ratio=%.2f%%',
                         len(self.is_solved),
                         np.mean(self.is_solved.astype(np.float)) * 100)
        self.rng = np.random.RandomState(42)
        self.balance_classes = balance_classes
        self.gen_hard_negatives = hard_negatives
        self.init_clip_ratio_to_sim = init_clip_ratio_to_sim
        self.init_frames_to_sim = init_frames_to_sim
        self.n_hist_frames = n_hist_frames
        self.frames_per_clip = frames_per_clip
        # If None, then set as initial clips to be sim, and hist frames
        self.frames_per_clip = (self.frames_per_clip or max(
            self.init_frames_to_sim, self.n_hist_frames))
        self.shuffle_indices = shuffle_indices
        self.drop_objs = drop_objs
        self.obj_fwd_model = obj_fwd_model


    def prep_drop_obs_list(self):
        drop_objs_lst = ()
        if not isinstance(self.drop_objs, int) and not self.drop_objs:
            # i.e. empty list, or None (and not an integer ID of obj to drop)
            pass
        elif isinstance(self.drop_objs, int):
            drop_objs_lst = (self.drop_objs, )
        elif isinstance(self.drop_objs, str):
            drop_objs_lst = (int(el) for el in self.drop_objs.split(';'))
        else:
            logging.warning('Not sure what was passed as drop objs %s',
                            self.drop_objs)
            drop_objs_lst = ()

        return drop_objs_lst


    def _gen_simulator(self):

        drop_objs_list = self.prep_drop_obs_list()
        simulator = phyre.initialize_simulator(self.task_ids,
                                               self.tier,
                                               drop_objs_list=drop_objs_list)
        phyre_sim = hydra.utils.instantiate(self.simulator_cfg, simulator, self.obj_fwd_model)
        return phyre_sim

    '''
    def encode_and_store_all(self):
        "Takes given task_ids, and action pairs and stores the in simulation output for viewership"
        phyre_sim = self._gen_simulator()
        actual_write_path = os.path.join(os.getcwd(), self.write_path)
        for each_idx in range(self.task_indices.shape[0]):
            res = {}
            res['batch_indices'] = each_idx
            res['task_indices'] = self.task_indices[each_idx]

            res['gtruth_task_id'] = self.task_ids[self.task_indices[each_idx]]
            res['action_taken'] = np.asarray(self.actions[each_idx], dtype=np.float32)

            res['vid_obs'], res['obj_obs'] =  phyre_sim.sim(
                [res['task_indices']], [res['action_taken']],
                self.init_clip_ratio_to_sim,
                nframes=self.frames_per_clip
            )
            print(type(res['vid_obs']))
            print(res['vid_obs'].shape)
            print(type(res['obj_obs']))
            print(res['obj_obs'].shape)



            break


    def _choose_negs(self, positives, negative_indices):
        normal_negatives = self.rng.choice(negative_indices, size=1)
        if self.rng.uniform() <= (1.0 - self.gen_hard_negatives):
            # Return the normal negative, else process to get hard one
            return normal_negatives
        positive_actions = self.actions[positives]
        # Find negatives as the closest action in euclidean space, that is
        # negative. It's slow, but pre-computing all distances is impossible
        # to store. So sub-sampling the negative indices to get a somewhat
        # coarser nearest element
        same_task_negative_indices = negative_indices[(
            self.task_indices[negative_indices] == self.task_indices[positives]
        )]
        # Just a cheap hack to compute the distances faster, selecting a subset
        # Ideally pre-compute the distances per-task, that might be storable
        # But this may not be too bad either
        negative_indices = self.rng.choice(same_task_negative_indices, 1000)
        negative_actions = self.actions[negative_indices]
        distances = scipy.spatial.distance.cdist(positive_actions,
                                                 negative_actions, 'euclidean')
        hard_neg_nearest = 1  # Randomly choose from this many
        nearest_neg_idx = negative_indices[np.argpartition(
            distances, hard_neg_nearest)[:, :hard_neg_nearest]]
        return np.stack([self.rng.choice(row) for row in nearest_neg_idx])

    '''

    def index_sampler(self):

        assert self.init_frames_to_sim == 0
        
        indices = np.arange(len(self.is_solved))
        
        if self.shuffle_indices:
            self.rng.shuffle(indices)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:

            per_worker = int(math.ceil(len(indices)// worker_info.num_workers))
            iter_start = worker_info.id * per_worker
            this_indices = indices[iter_start :
                                   min(iter_start + per_worker, len(indices))]
            this_is_solved = self.is_solved[iter_start:
                                            min(iter_start + per_worker, len(indices))]

        else:
            this_indices = indices
            this_is_solved = self.is_solved

        if self.balance_classes:
            solved_mask = this_is_solved > 0
            positive_indices = this_indices[solved_mask]
            negative_indices = this_indices[~solved_mask]
            while True:
                positives = self.rng.choice(positive_indices, size=1)
                negatives = self._choose_negs(positives, negative_indices)
                yield np.concatenate([positives, negatives])

        else:
            while True:
                yield self.rng.choice(this_indices, size=2)


    def iter_indices(self):

        phyre_sim = self._gen_simulator()
        for ids in self.index_sampler():
            res = {}
            res['batch_indices'] = ids
            res['task_indices'] = self.task_indices[ids]
            res['actions'] = self.actions[ids]
            res['is_solved'] = self.is_solved[ids]
            
            res['vid_obs'], res['obj_obs'] = phyre_sim.sim(
                res['task_indices'],
                res['actions'],
                self.init_clip_ratio_to_sim,
                nframes=self.frames_per_clip
            )
            
            res['vid_obs'] = np.array(res['vid_obs'], dtype=np.long)
            res['obj_obs'] = np.array(res['obj_obs'], dtype=np.float32)
            yield res



    def __iter__(self):

        self.iter_indices()


    def read_video(self):

        pass


    def __getitem__(self):
        pass


    def __len__(self):
        pass


class PhyreDataset(torch.utils.data.IterableDataset):
    """Phyre Dataset iterable."""
    def __init__(self,
                 tier,
                 task_ids,
                 task_indices,
                 is_solved,
                 actions,
                 simulator_cfg,
                 mode='train',
                 balance_classes=True,
                 hard_negatives=0.0,
                 init_clip_ratio_to_sim=-1,
                 init_frames_to_sim=0,
                 frames_per_clip=1,
                 n_hist_frames=3,
                 shuffle_indices=False,
                 drop_objs='',
                 obj_fwd_model=None):
        """
        Args:
            task_indices: The integer indices into the simulator tasks (not the
                actual task IDs used by the simulator).
            hard_negatives (float): The ratio of times to find a hard negative
                instead of a normal negative. Hard negatives are close to the
            shuffle_indices (bool): Shuffle the indices that each worker
                gets. Setting to false by default since that's how most initial
                models were trained, though it leads to batches having a certain
                set of templates only (rather than a uniform mix), since each
                worker gets a uniform set of templates to load.
            drop_objs: (str): ';' separated list of objects to be dropped.
        """
        self.tier = tier
        self.task_ids = np.array(task_ids)
        self.task_indices = np.array(task_indices)
        self.is_solved = np.array(is_solved)
        self.actions = (actions.cpu().numpy()
                        if torch.is_tensor(actions) else np.array(actions))
        self.simulator_cfg = simulator_cfg
        self.mode = mode
        if self.mode == 'train':
            logging.info('Data set: size=%d, positive_ratio=%.2f%%',
                         len(self.is_solved),
                         np.mean(self.is_solved.astype(np.float)) * 100)
        self.rng = np.random.RandomState(42)
        self.balance_classes = balance_classes
        self.gen_hard_negatives = hard_negatives
        self.init_clip_ratio_to_sim = init_clip_ratio_to_sim
        self.init_frames_to_sim = init_frames_to_sim
        self.n_hist_frames = n_hist_frames
        self.frames_per_clip = frames_per_clip
        # If None, then set as initial clips to be sim, and hist frames
        self.frames_per_clip = (self.frames_per_clip or max(
            self.init_frames_to_sim, self.n_hist_frames))
        self.shuffle_indices = shuffle_indices
        self.drop_objs = drop_objs
        self.obj_fwd_model = obj_fwd_model

    def _gen_simulator(self):
        drop_objs_lst = ()
        if not isinstance(self.drop_objs, int) and not self.drop_objs:
            # i.e. empty list, or None (and not an integer ID of obj to drop)
            pass
        elif isinstance(self.drop_objs, int):
            drop_objs_lst = (self.drop_objs, )
        elif isinstance(self.drop_objs, str):
            drop_objs_lst = (int(el) for el in self.drop_objs.split(';'))
        else:
            logging.warning('Not sure what was passed as drop objs %s',
                            self.drop_objs)
            drop_objs_lst = ()
        simulator = phyre.initialize_simulator(self.task_ids,
                                               self.tier,
                                               drop_objs=drop_objs_lst)
        phyre_sim = hydra.utils.instantiate(self.simulator_cfg, simulator, self.obj_fwd_model)
        return phyre_sim

    def _choose_negs(self, positives, negative_indices):
        normal_negatives = self.rng.choice(negative_indices, size=1)
        if self.rng.uniform() <= (1.0 - self.gen_hard_negatives):
            # Return the normal negative, else process to get hard one
            return normal_negatives
        positive_actions = self.actions[positives]
        # Find negatives as the closest action in euclidean space, that is
        # negative. It's slow, but pre-computing all distances is impossible
        # to store. So sub-sampling the negative indices to get a somewhat
        # coarser nearest element
        same_task_negative_indices = negative_indices[(
            self.task_indices[negative_indices] == self.task_indices[positives]
        )]
        # Just a cheap hack to compute the distances faster, selecting a subset
        # Ideally pre-compute the distances per-task, that might be storable
        # But this may not be too bad either
        negative_indices = self.rng.choice(same_task_negative_indices, 1000)
        negative_actions = self.actions[negative_indices]
        distances = scipy.spatial.distance.cdist(positive_actions,
                                                 negative_actions, 'euclidean')
        hard_neg_nearest = 1  # Randomly choose from this many
        nearest_neg_idx = negative_indices[np.argpartition(
            distances, hard_neg_nearest)[:, :hard_neg_nearest]]
        return np.stack([self.rng.choice(row) for row in nearest_neg_idx])

    def _train_indices_sampler(self):
        """Returns a pair of IDs, balanced if asked for."""
        # Pair so that if asked for balanced, it will return a postiive and
        # negative, hence satisfying the constraint
        assert self.init_frames_to_sim == 0, 'Not handled here'
        indices = np.arange(len(self.is_solved))
        if self.shuffle_indices:
            self.rng.shuffle(indices)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # split workload, multiple workers working on the data
            per_worker = int(math.ceil(len(indices) / worker_info.num_workers))
            iter_start = worker_info.id * per_worker
            this_indices = indices[iter_start:min(iter_start +
                                                  per_worker, len(indices))]
            this_is_solved = self.is_solved[iter_start:min(
                iter_start + per_worker, len(indices))]
        else:
            this_indices = indices
            this_is_solved = self.is_solved
        if self.balance_classes:
            solved_mask = this_is_solved > 0
            positive_indices = this_indices[solved_mask]
            negative_indices = this_indices[~solved_mask]
            while True:
                positives = self.rng.choice(positive_indices, size=1)
                negatives = self._choose_negs(positives, negative_indices)
                yield np.concatenate((positives, negatives))
        else:
            while True:
                yield self.rng.choice(this_indices, size=2)

    def _iter_train(self):
        phyre_sim = self._gen_simulator()
        for ids in self._train_indices_sampler():
            res = {}
            res['batch_indices'] = ids
            res['task_indices'] = self.task_indices[ids]
            res['actions'] = self.actions[ids]
            res['is_solved'] = self.is_solved[ids]
            # Get the actual video by simulation
            res['vid_obs'], res['obj_obs'] = phyre_sim.sim(
                res['task_indices'],
                res['actions'],
                self.init_clip_ratio_to_sim,
                nframes=self.frames_per_clip)
            res['vid_obs'] = np.array(res['vid_obs'], dtype=np.long)
            res['obj_obs'] = np.array(res['obj_obs'], dtype=np.float32)
            yield res

    def _test_indices_sampler(self):
        """Just run in order of actions, need to eval all."""
        indices = np.arange(len(self.task_indices))
        assert self.shuffle_indices is False, 'No good reason shuffle for test'
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # split workload, multiple workers working on the data
            per_worker = int(math.ceil(len(indices) / worker_info.num_workers))
            iter_start = worker_info.id * per_worker
            this_indices = indices[iter_start:min(iter_start +
                                                  per_worker, len(indices))]
        else:
            this_indices = indices
        for index in this_indices:
            yield index

    def _iter_test(self):
        phyre_sim = self._gen_simulator()
        for ids in self._test_indices_sampler():
            res = {}
            res['batch_indices'] = ids
            res['task_indices'] = self.task_indices[ids]
            res['task_ids'] = self.task_ids[self.task_indices[ids]]
            res['actions'] = np.array(self.actions[ids], dtype=np.float32)
            # Get the actual video by simulation
            # Get the actual video obs
            vid_obs_orig, obj_obs_orig = phyre_sim.sim(
                [res['task_indices']], [res['actions']],
                self.init_clip_ratio_to_sim,
                nframes=self.frames_per_clip)
            # Use the portion of frames
            vid_obs = [
                el[:min(el.shape[0], self.init_frames_to_sim)]
                for el in vid_obs_orig
            ]
            obj_obs = [
                el[:min(el.shape[0], self.init_frames_to_sim)]
                for el in obj_obs_orig
            ]
            vid_obs = [el[-self.n_hist_frames:] for el in vid_obs]
            obj_obs = [el[-self.n_hist_frames:] for el in obj_obs]
            # If the length is less than n_hist_frames, then pad with empty
            # frames at the beginning
            pad_len = self.n_hist_frames - vid_obs[0].shape[0]
            if pad_len > 0:
                logging.debug('Appending empty frames since n_hist_frames > '
                              'frames returned from the reader')
                vid_obs = [
                    np.concatenate([np.zeros_like(el[:1])] * pad_len + [el],
                                   axis=0) for el in vid_obs
                ]
                obj_obs = [
                    np.concatenate([np.zeros_like(el[:1])] * pad_len + [el],
                                   axis=0) for el in obj_obs
                ]
            vid_obs = np.array(vid_obs, dtype=np.long)
            obj_obs = np.array(obj_obs, dtype=np.float32)
            # Check that batch size and number of frames is same for obj/vid
            assert np.shape(vid_obs)[1] == self.n_hist_frames
            assert np.shape(obj_obs)[1] == self.n_hist_frames
            assert np.shape(obj_obs)[0] == np.shape(vid_obs)[0]
            res['vid_obs'] = vid_obs
            res['vid_obs_orig'] = vid_obs_orig
            res['vid_obs'] = np.array(res['vid_obs'], dtype=np.long)
            res['vid_obs_orig'] = np.array(res['vid_obs_orig'], dtype=np.long)
            res['obj_obs'] = obj_obs
            res['obj_obs_orig'] = obj_obs_orig
            res['obj_obs'] = np.array(res['obj_obs'], dtype=np.float32)
            res['obj_obs_orig'] = np.array(res['obj_obs_orig'],
                                           dtype=np.float32)
            yield res

    def __iter__(self):
        if self.mode == 'train':
            return self._iter_train()
        elif self.mode == 'test':
            return self._iter_test()
        else:
            raise NotImplementedError('Unknown {}'.format(self.mode))


@hydra.main(config_path='conf/config.yaml')
def test_dataset(cfg):
    from train2 import get_train_test, get_subset_tasks, find_all_agents
    from functools import partial
    import neural_agent

    train_task_ids, eval_task_ids = get_train_test(cfg.eval_setup_name,
                                                   cfg.fold_id,
                                                   cfg.use_test_split)

    train_task_ids = sorted(train_task_ids)
    eval_task_ids = sorted(eval_task_ids)

    train_task_ids = get_subset_tasks(train_task_ids, cfg.data_ratio_train)
    eval_task_ids = get_subset_tasks(eval_task_ids, cfg.data_ratio_eval)

    cfg.tier = phyre.eval_setup_to_action_tier(cfg.eval_setup_name)


    cache = phyre.get_default_100k_cache(cfg.tier)
    training_data = cache.get_sample(train_task_ids, cfg.train.data_loader.max_train_actions)
    task_indices, is_solved, actions, _, _ = neural_agent.compact_simulation_data_to_trainset(cfg.tier, **training_data)

    dataset = PhyreDataset(
        cfg.tier,
        train_task_ids,
        task_indices,
        is_solved,
        actions,
        cfg.simulator,
        mode='train',
        balance_classes=cfg.train.data_loader.balance_classes,  # True
        hard_negatives=cfg.train.data_loader.hard_negatives,  # 0.0
        init_clip_ratio_to_sim=cfg.train.init_clip_ratio_to_sim,  # -1
        frames_per_clip=cfg.train.frames_per_clip,  # 4
        n_hist_frames=cfg.train.n_hist_frames,  # 3
        shuffle_indices=cfg.train.shuffle_indices,  # False
        drop_objs=cfg.train.drop_objs,  # ''
        obj_fwd_model=None
    )

    phyre_sim = dataset._gen_simulator()

    for ids in dataset._train_indices_sampler():

        res = {}
        res['task_indices'] = dataset.task_indices[ids]
        res['actions'] = dataset.actions[ids]

        res['vid_obs'], res['obj_obs'] = phyre_sim.sim(
            res['task_indices'],
            res['actions'],
            dataset.init_clip_ratio_to_sim,
            nframes=dataset.frames_per_clip)

@hydra.main(config_path='conf_dataset/config.yaml')
def encode_dataset(cfg):

    from train2 import get_train_test
    import neural_agent

    train_task_ids, eval_task_ids = get_train_test(cfg.eval_setup_name,
                                                   cfg.fold_id,
                                                   cfg.use_test_split)

    all_task_ids = train_task_ids + eval_task_ids
    all_task_ids = sorted(all_task_ids)

    cfg.tier = phyre.eval_setup_to_action_tier(cfg.eval_setup_name)
    cache = phyre.get_default_100k_cache(cfg.tier)
    all_data = cache.get_sample(all_task_ids, cfg.train.data_loader.max_train_actions)
    task_indices, is_solved, actions, _, _ = neural_agent.compact_simulation_data_to_trainset(cfg.tier, **all_data)


    write_path = '/proj/vondrick/ishaan/datasets/phyre'
    if not os.path.exists(write_path):
        os.makedirs(write_path)

    dataset = PhyreVideoSavedDataset(
        write_path,
        cfg.tier,
        all_task_ids,
        task_indices,
        is_solved,
        actions,
        cfg.simulator,
        mode='train',
        balance_classes=cfg.train.data_loader.balance_classes,  # True
        hard_negatives=cfg.train.data_loader.hard_negatives,  # 0.0
        init_clip_ratio_to_sim=cfg.train.init_clip_ratio_to_sim,  # -1
        frames_per_clip=cfg.train.frames_per_clip,  # 4
        n_hist_frames=cfg.train.n_hist_frames,  # 3
        shuffle_indices=cfg.train.shuffle_indices,  # False
        drop_objs=cfg.train.drop_objs,  # ''
        obj_fwd_model=None
    )

    for batch_data_id, batch_data in enumerate(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=im_fwd_agent.get_num_workers(
                    cfg.train.data_loader.num_workers,
                    dataset.frames_per_clip),
                pin_memory=False,
                # Asking for half the batch size since the dataloader is designed
                # to give 2 elements per batch (for class balancing)
                batch_size= cfg.train_batch_size // 2)):


        print(batch_data)






if __name__ == '__main__':

    encode_dataset()