import importlib
import pickle
import time
import numpy
import torch
import math
import ray
import copy
import pathlib
from algorithm.muzero import  replay_buffer, self_play, shared_storage, trainer

from torch.utils.tensorboard import SummaryWriter

import tqdm


class MuZero:
    def __init__(self,game_name, config=None, split_resources_in=1) -> None:
        # Load the game and the config from the module with the game name
        try:
            game_module = importlib.import_module("games." + game_name)
            self.Game = game_module.Game
            self.config = game_module.MuZeroConfig()
        except ModuleNotFoundError as err:
            print(
                f'{game_name} is not a supported game name, try "cartpole" or refer to the documentation for adding a new game.'
            )
            raise err
        # 检查配置Overwrite the config
        if config:
            if type(config) is dict:
                for param, value in config.items():
                    if hasattr(self.config, param):
                        setattr(self.config, param, value)
                    else:
                        raise AttributeError(
                            f"{game_name} config has no attribute '{param}'. Check the config file for the complete list of parameters."
                        )
            else:
                self.config = config
        
        # 固定随机种子 Fix random generator seed
        numpy.random.seed(self.config.seed)
        torch.manual_seed(self.config.seed)


        # Manage GPUs
        # if self.config.max_num_gpus == 0 and (
        #     self.config.selfplay_on_gpu
        #     or self.config.train_on_gpu
        #     or self.config.reanalyse_on_gpu
        # ):
        #     raise ValueError(
        #         "Inconsistent MuZeroConfig: max_num_gpus = 0 but GPU requested by selfplay_on_gpu or train_on_gpu or reanalyse_on_gpu."
        #     )
        # if (
        #     self.config.selfplay_on_gpu
        #     or self.config.train_on_gpu
        #     or self.config.reanalyse_on_gpu
        # ):
        #     total_gpus = (
        #         self.config.max_num_gpus
        #         if self.config.max_num_gpus is not None
        #         else torch.cuda.device_count()
        #     )
        # else:
        #     total_gpus = 0

        # self.num_gpus = total_gpus / split_resources_in
        # if 1 < self.num_gpus:
        #     self.num_gpus = math.floor(self.num_gpus)
        if torch.cuda.is_available():
            self.config.selfplay_on_gpu = True
            self.config.train_on_gpu = True
        else:
            self.config.selfplay_on_gpu = False
            self.config.train_on_gpu = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
        # print(self.config.selfplay_on_gpu)

        # ray.init(num_gpus=total_gpus, ignore_reinit_error=True)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }
        self.replay_buffer = {}
        
        # config_copy = copy.deepcopy(self.config)
        # config_ref = ray.put(config_copy)
        # cpu_actor = CPUActor.remote()
        # cpu_weights = cpu_actor.get_initial_weights.remote(config_ref)
        # self.checkpoint["weights"], self.summary = copy.deepcopy(ray.get(cpu_weights))
        # Initialize CPUActor, SharedStorage, ReplayBuffer, and Trainer
        self.cpu_actor = CPUActor()
        self.checkpoint["weights"], self.summary = self.cpu_actor.get_initial_weights(
            copy.deepcopy(self.config)
        )
        self.shared_storage = shared_storage.SharedStorage(self.checkpoint, self.config)
        self.replay_buffer_worker = replay_buffer.ReplayBuffer(self.checkpoint, self.replay_buffer, self.config)
        self.training_worker = trainer.Trainer(self.checkpoint, self.config)

        # Self-play workers
        self.self_play_workers = [
            self_play.SelfPlay(self.checkpoint, self.Game, self.config, self.config.seed + seed)
            for seed in range(self.config.num_workers)
        ]
        self.writer = SummaryWriter(self.config.results_path)



         # Workers
        # self.self_play_workers = None
        # self.test_worker = None
        # self.training_worker = None
        # self.reanalyse_worker = None
        # self.replay_buffer_worker = None
        # self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        """
        Spawn ray workers and launch the training.

        Args:
            log_in_tensorboard (bool): Start a testing worker and log its performance in TensorBoard.
        """
        self.train_step = 0
        if log_in_tensorboard or self.config.save_model:
            self.config.results_path.mkdir(parents=True, exist_ok=True)

        # self.shared_storage_worker.set_info("terminate", False)

        # Run self-play loop for each worker
        for self.train_step in tqdm.tqdm(range(self.config.training_steps)):
            for self_play_worker in self.self_play_workers:
                # print(self_play_worker)
                self_play_worker.self_play(self.shared_storage, self.replay_buffer_worker,  self.writer, self.train_step,num=2)

            # Run training loop
            self.training_worker.one_update_weights(self.replay_buffer_worker, self.shared_storage, self.writer, self.train_step,num=3)


            self_play_worker.self_play(self.shared_storage, None,  self.writer, self.train_step,test_mode=True)
            self.logging()



    def logging(self):
        # print("Logging")
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = self.shared_storage.get_info(keys)
        # print(info)
        print(f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}')
        self.writer.add_scalar("1.Total_reward/1.Total_reward", info["total_reward"], self.train_step)
        self.writer.add_scalar("1.Total_reward/2.Mean_value", info["mean_value"], self.train_step)
        self.writer.add_scalar("1.Total_reward/3.Episode_length", info["episode_length"], self.train_step)
        self.writer.add_scalar("1.Total_reward/4.MuZero_reward", info["muzero_reward"], self.train_step)
        self.writer.add_scalar("1.Total_reward/5.Opponent_reward", info["opponent_reward"], self.train_step)
        self.writer.add_scalar("2.Workers/1.Self_played_games", info["num_played_games"], self.train_step)
        self.writer.add_scalar("2.Workers/2.Training_steps", info["training_step"], self.train_step)
        self.writer.add_scalar("2.Workers/3.Self_played_steps", info["num_played_steps"], self.train_step)
        self.writer.add_scalar("2.Workers/4.Reanalysed_games", info["num_reanalysed_games"], self.train_step)
        self.writer.add_scalar("2.Workers/5.Training_steps_per_self_played_step_ratio", info["training_step"] / max(1, info["num_played_steps"]), self.train_step)
        self.writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], self.train_step)
        self.writer.add_scalar("3.Loss/1.Total_weighted_loss", info["total_loss"], self.train_step)
        self.writer.add_scalar("3.Loss/Value_loss", info["value_loss"], self.train_step)
        self.writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], self.train_step)
        self.writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], self.train_step)

    
    def test(self, render=False, opponent=None, muzero_player=None, num_tests=1, num_gpus=0):
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))
        results = []
        for i in range(num_tests):
            # print(f"Testing {i+1}/{num_tests}")
            results.append(
                self_play_worker.play_game(
                    temperature=0,
                    temperature_threshold=0,
                    render=render,
                    opponent=opponent,
                    muzero_player=muzero_player,
                )
            )
        self_play_worker.close_game()
        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result
        


        
        

        # Run logging if requested
        # if log_in_tensorboard:
        #     self.logging_loop()

        # if log_in_tensorboard or self.config.save_model:
        #     self.config.results_path.mkdir(parents=True, exist_ok=True)
        
        # Manage GPUs
        # if 0 < self.num_gpus:
        #     num_gpus_per_worker = self.num_gpus / (
        #         self.config.train_on_gpu
        #         + self.config.num_workers * self.config.selfplay_on_gpu
        #         + log_in_tensorboard * self.config.selfplay_on_gpu
        #         + self.config.use_last_model_value * self.config.reanalyse_on_gpu
        #     )
        #     if 1 < num_gpus_per_worker:
        #         num_gpus_per_worker = math.floor(num_gpus_per_worker)
        # else:
        #     num_gpus_per_worker = 0

        # # Initialize workers
        # self.training_worker = trainer.Trainer.options(
        #     num_cpus=0,
        #     num_gpus=num_gpus_per_worker if self.config.train_on_gpu else 0,
        # ).remote(self.checkpoint, self.config)

        # self.shared_storage_worker = shared_storage.SharedStorage.remote(
        #     self.checkpoint,
        #     self.config,
        # )
        # self.shared_storage_worker.set_info.remote("terminate", False)

        # self.replay_buffer_worker = replay_buffer.ReplayBuffer.remote(
        #     self.checkpoint, self.replay_buffer, self.config
        # )

        # if self.config.use_last_model_value:
        #     self.reanalyse_worker = replay_buffer.Reanalyse.options(
        #         num_cpus=0,
        #         num_gpus=num_gpus_per_worker if self.config.reanalyse_on_gpu else 0,
        #     ).remote(self.checkpoint, self.config)
        
        # self.self_play_workers = [
        #     self_play.SelfPlay.options(
        #         num_cpus=0,
        #         num_gpus=num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
        #     ).remote(
        #         self.checkpoint,
        #         self.Game,
        #         self.config,
        #         self.config.seed + seed,
        #     )
        #     for seed in range(self.config.num_workers)
        # ]

        # # Launch workers
        # [
        #     self_play_worker.continuous_self_play.remote(
        #         self.shared_storage_worker, self.replay_buffer_worker
        #     )
        #     for self_play_worker in self.self_play_workers
        # ]
        # self.training_worker.continuous_update_weights.remote(
        #     self.replay_buffer_worker, self.shared_storage_worker
        # )
        # if self.config.use_last_model_value:
        #     self.reanalyse_worker.reanalyse.remote(
        #         self.replay_buffer_worker, self.shared_storage_worker
        #     )

        # if log_in_tensorboard:
        #     self.logging_loop(
        #         num_gpus_per_worker if self.config.selfplay_on_gpu else 0,
        #     )
        
    def logging_loop(self):
        """
        Keep track of the training performance.
        """
        # Initialize TensorBoard
        writer = SummaryWriter(self.config.results_path)

        print(
            "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
        )

        # Save hyperparameters to TensorBoard
        hp_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text(
            "Hyperparameters",
            "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
        )

        # Save model summary
        writer.add_text("Model summary", self.summary)

        # Logging loop
        counter = 0
        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        try:
            while self.shared_storage.get_info("training_step") < self.config.training_steps:
                info = self.shared_storage.get_info(keys)
                writer.add_scalar("1.Total_reward/1.Total_reward", info["total_reward"], counter)
                writer.add_scalar("1.Total_reward/2.Mean_value", info["mean_value"], counter)
                # Add more scalars as needed...
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        self.terminate_workers()

        if self.config.save_model:
            # Persist replay buffer to disk
            path = self.config.results_path / "replay_buffer.pkl"
            print(f"\n\nPersisting replay buffer games to disk at {path}")
            with open(path, "wb") as f:
                pickle.dump(
                    {
                        "buffer": self.replay_buffer,
                        "num_played_games": self.checkpoint["num_played_games"],
                        "num_played_steps": self.checkpoint["num_played_steps"],
                        "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
                    },
                    f,
                )

        # Launch the test worker to get performance metrics
    #     self.test_worker = self_play.SelfPlay.options(
    #         num_cpus=0,
    #         num_gpus=num_gpus,
    #     ).remote(
    #         self.checkpoint,
    #         self.Game,
    #         self.config,
    #         self.config.seed + self.config.num_workers,
    #     )
    #     self.test_worker.continuous_self_play.remote(
    #         self.shared_storage_worker, None, True
    #     )

    #     # Write everything in TensorBoard
    #     writer = SummaryWriter(self.config.results_path)

    #     print(
    #         "\nTraining...\nRun tensorboard --logdir ./results and go to http://localhost:6006/ to see in real time the training performance.\n"
    #     )

    #     # Save hyperparameters to TensorBoard
    #     hp_table = [
    #         f"| {key} | {value} |" for key, value in self.config.__dict__.items()
    #     ]
    #     writer.add_text(
    #         "Hyperparameters",
    #         "| Parameter | Value |\n|-------|-------|\n" + "\n".join(hp_table),
    #     )
    #     # Save model representation
    #     writer.add_text(
    #         "Model summary",
    #         self.summary,
    #     )
    #     # Loop for updating the training performance
    #     counter = 0
    #     keys = [
    #         "total_reward",
    #         "muzero_reward",
    #         "opponent_reward",
    #         "episode_length",
    #         "mean_value",
    #         "training_step",
    #         "lr",
    #         "total_loss",
    #         "value_loss",
    #         "reward_loss",
    #         "policy_loss",
    #         "num_played_games",
    #         "num_played_steps",
    #         "num_reanalysed_games",
    #     ]
    #     info = ray.get(self.shared_storage_worker.get_info.remote(keys))
    #     try:
    #         while info["training_step"] < self.config.training_steps:
    #             info = ray.get(self.shared_storage_worker.get_info.remote(keys))
    #             writer.add_scalar(
    #                 "1.Total_reward/1.Total_reward",
    #                 info["total_reward"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "1.Total_reward/2.Mean_value",
    #                 info["mean_value"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "1.Total_reward/3.Episode_length",
    #                 info["episode_length"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "1.Total_reward/4.MuZero_reward",
    #                 info["muzero_reward"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "1.Total_reward/5.Opponent_reward",
    #                 info["opponent_reward"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "2.Workers/1.Self_played_games",
    #                 info["num_played_games"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "2.Workers/2.Training_steps", info["training_step"], counter
    #             )
    #             writer.add_scalar(
    #                 "2.Workers/3.Self_played_steps", info["num_played_steps"], counter
    #             )
    #             writer.add_scalar(
    #                 "2.Workers/4.Reanalysed_games",
    #                 info["num_reanalysed_games"],
    #                 counter,
    #             )
    #             writer.add_scalar(
    #                 "2.Workers/5.Training_steps_per_self_played_step_ratio",
    #                 info["training_step"] / max(1, info["num_played_steps"]),
    #                 counter,
    #             )
    #             writer.add_scalar("2.Workers/6.Learning_rate", info["lr"], counter)
    #             writer.add_scalar(
    #                 "3.Loss/1.Total_weighted_loss", info["total_loss"], counter
    #             )
    #             writer.add_scalar("3.Loss/Value_loss", info["value_loss"], counter)
    #             writer.add_scalar("3.Loss/Reward_loss", info["reward_loss"], counter)
    #             writer.add_scalar("3.Loss/Policy_loss", info["policy_loss"], counter)
    #             print(
        #                 f'Last test reward: {info["total_reward"]:.2f}. Training step: {info["training_step"]}/{self.config.training_steps}. Played games: {info["num_played_games"]}. Loss: {info["total_loss"]:.2f}',
        #                 end="\r",
    #             )
    #             counter += 1
    #             time.sleep(0.5)
    #     except KeyboardInterrupt:
    #         pass

    #     self.terminate_workers()

    #     if self.config.save_model:
    #         # Persist replay buffer to disk
    #         path = self.config.results_path / "replay_buffer.pkl"
    #         print(f"\n\nPersisting replay buffer games to disk at {path}")
    #         pickle.dump(
    #             {
    #                 "buffer": self.replay_buffer,
    #                 "num_played_games": self.checkpoint["num_played_games"],
    #                 "num_played_steps": self.checkpoint["num_played_steps"],
    #                 "num_reanalysed_games": self.checkpoint["num_reanalysed_games"],
    #             },
    #             open(path, "wb"),
    #         )

    def terminate_workers(self):
        """
        Softly terminate the running tasks and garbage collect the workers.
        """
        self.shared_storage.set_info("terminate", True)
        self.checkpoint = self.shared_storage.get_checkpoint()
        self.replay_buffer = self.replay_buffer_worker.get_buffer()
        print("\nShutting down workers...")
        # if self.shared_storage_worker:
        #     self.shared_storage_worker.set_info.remote("terminate", True)
        #     self.checkpoint = ray.get(
        #         self.shared_storage_worker.get_checkpoint.remote()
        #     )
        # if self.replay_buffer_worker:
        #     self.replay_buffer = ray.get(self.replay_buffer_worker.get_buffer.remote())

        # print("\nShutting down workers...")

        # self.self_play_workers = None
        # self.test_worker = None
        # self.training_worker = None
        # self.reanalyse_worker = None
        # self.replay_buffer_worker = None
        # self.shared_storage_worker = None

    def test(
        self, render=True, opponent=None, muzero_player=None, num_tests=1, num_gpus=0
    ):
        """
        Test the model in a dedicated thread.

        Args:
            render (bool): To display or not the environment. Defaults to True.

            opponent (str): "self" for self-play, "human" for playing against MuZero and "random"
            for a random agent, None will use the opponent in the config. Defaults to None.

            muzero_player (int): Player number of MuZero in case of multiplayer
            games, None let MuZero play all players turn by turn, None will use muzero_player in
            the config. Defaults to None.

            num_tests (int): Number of games to average. Defaults to 1.

            num_gpus (int): Number of GPUs to use, 0 forces to use the CPU. Defaults to 0.
        """
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_play_worker = self_play.SelfPlay.options(
            num_cpus=0,
            num_gpus=num_gpus,
        ).remote(self.checkpoint, self.Game, self.config, numpy.random.randint(10000))

        results = []
        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(
                self_play_worker.play_game(
                    temperature=0,
                    temperature_threshold=self.config.temperature_threshold,
                    render=render,
                    opponent=opponent,
                    muzero_player=muzero_player,
                )
            )
        self_play_worker.close_game.remote()

        if len(self.config.players) == 1:
            result = numpy.mean([sum(history.reward_history) for history in results])
        else:
            result = numpy.mean(
                [
                    sum(
                        reward
                        for i, reward in enumerate(history.reward_history)
                        if history.to_play_history[i - 1] == muzero_player
                    )
                    for history in results
                ]
            )
        return result

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        """
        Load a model and/or a saved replay buffer.

        Args:
            checkpoint_path (str): Path to model.checkpoint or model.weights.

            replay_buffer_path (str): Path to replay_buffer.pkl
        """
        # Load checkpoint
        if checkpoint_path:
            checkpoint_path = pathlib.Path(checkpoint_path)
            self.checkpoint = torch.load(checkpoint_path)
            print(f"\nUsing checkpoint from {checkpoint_path}")

        # Load replay buffer
        if replay_buffer_path:
            replay_buffer_path = pathlib.Path(replay_buffer_path)
            with open(replay_buffer_path, "rb") as f:
                replay_buffer_infos = pickle.load(f)
            self.replay_buffer = replay_buffer_infos["buffer"]
            self.checkpoint["num_played_steps"] = replay_buffer_infos[
                "num_played_steps"
            ]
            self.checkpoint["num_played_games"] = replay_buffer_infos[
                "num_played_games"
            ]
            self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                "num_reanalysed_games"
            ]

            print(f"\nInitializing replay buffer with {replay_buffer_path}")
        else:
            print(f"Using empty buffer.")
            self.replay_buffer = {}
            self.checkpoint["training_step"] = 0
            self.checkpoint["num_played_steps"] = 0
            self.checkpoint["num_played_games"] = 0
            self.checkpoint["num_reanalysed_games"] = 0

    # def diagnose_model(self, horizon):
    #     """
    #     Play a game only with the learned model then play the same trajectory in the real
    #     environment and display information.

    #     Args:
    #         horizon (int): Number of timesteps for which we collect information.
    #     """
    #     game = self.Game(self.config.seed)
    #     obs = game.reset()
    #     dm = diagnose_model.DiagnoseModel(self.checkpoint, self.config)
    #     dm.compare_virtual_with_real_trajectories(obs, game, horizon)
    #     input("Press enter to close all plots")
    #     dm.close_all()


class CPUActor:
    # Trick to force DataParallel to stay on CPU to get weights on CPU even if there is a GPU
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        from algorithm.muzero import models
        # config = ray.get(config_ref)
        model = models.MuZeroNetwork(config)
        weigths = model.get_weights()
        summary = str(model).replace("\n", " \n\n")
        return weigths, summary
