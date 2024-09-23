from stable_baselines3 import PPO, A2C
import os
from WukongEnv import WukongEnv
from stable_baselines3.common.callbacks import BaseCallback
class DetailedLossCallback(BaseCallback):
	def __init__(self, verbose=0):
		super(DetailedLossCallback, self).__init__(verbose)
		self.policy_losses = []
		self.value_losses = []

	def _on_step(self) -> bool:
		if self.n_calls % 100 == 0:  # 每100步打印一次
			if hasattr(self.model, 'logger'):
				logs = self.model.logger.name_to_value
				if 'loss/policy_loss' in logs and 'loss/value_loss' in logs:
					policy_loss = logs['loss/policy_loss']
					value_loss = logs['loss/value_loss']
					print(f"Step: {self.n_calls}")
					print(f"  Policy Loss: {policy_loss}")
					print(f"  Value Loss: {value_loss}")
					self.policy_losses.append(policy_loss)
					self.value_losses.append(value_loss)
		return True

	def on_training_end(self) -> None:
		import matplotlib.pyplot as plt
		plt.figure(figsize=(10, 5))
		plt.plot(self.policy_losses, label='Policy Loss')
		plt.plot(self.value_losses, label='Value Loss')
		plt.legend()
		plt.title('Policy and Value Losses')
		plt.xlabel('Updates (x100 steps)')
		plt.ylabel('Loss')
		plt.savefig('loss_plot.png')
		plt.close()
def train(CREATE_NEW_MODEL, config):
	print("🧠 Training will start soon. This can take a while to initialize...")


	TIMESTEPS = 1			#Learning rate multiplier.
	HORIZON_WINDOW = 500	#Lerning rate number of steps before updating the model. ~2min


	'''Creating folder structure'''
	model_name = "PPO-1"#Your name here
	if not os.path.exists(f"models/{model_name}/"):
		os.makedirs(f"models/{model_name}/")
	if not os.path.exists(f"logs/{model_name}/"):
		os.makedirs(f"logs/{model_name}/")
	models_dir = f"models/{model_name}/"
	logdir = f"logs/{model_name}/"			
	model_path = f"{models_dir}/PPO-1"
	print("🧠 Folder structure created...")

	config["logdir"] = logdir
	'''Initializing environment'''
	env = WukongEnv(config)
	print("🧠 WukongEnv initialized...")



	'''Creating new model or loading existing model'''
	if CREATE_NEW_MODEL:
		model = PPO('MultiInputPolicy',
							env,
							tensorboard_log=logdir,
							n_steps=HORIZON_WINDOW,
							verbose=1,
							device='cpu')	#Set training device here.
		print("🧠 New Model created...")
	else:
		model = PPO.load(model_path, env=env)
		print("🧠 Model loaded...")

	callback = DetailedLossCallback()
	'''Training loop'''
	while True:
		model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO", log_interval=1,
					callback=callback)
		model.save(f"{models_dir}/PPO-1")
		print(f"🧠 Model updated...")