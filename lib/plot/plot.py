
# W1D2_Tutorial1.ipynb

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

def plot_single_model_train_metrics(value_dict):
	# Create a single plot for all training costs with a logarithmic scale
	with plt.xkcd():
		plt.figure(figsize=(8, 6))  # Set the figure size
		####
		colour = {'train':'tab:red','val':'tab:blue','test':'tab:green'}
		
		# Create some mock data
		t = range(1, len(value_dict['train_acc'])+1)

		fig, ax1 = plt.subplots()

		ax1.set_xlabel('epoch')
		ax1.set_ylabel('acc')# ax1.set_ylabel('acc', color=color)
		for key, val in colour.items():
			ax1.plot(t, value_dict[key+'_acc'], color=val, linestyle='-', label=key+'_acc')

		ax1.tick_params(axis='y') # ax1.tick_params(axis='y', labelcolor=color)


		ax2 = ax1.twinx()  # instantiate a second Axes that shares the same x-axis
		ax2.set_ylabel('loss')  # we already handled the x-label with ax1
		for key, val in colour.items():
			ax2.plot(t, value_dict[key+'_loss'], color=val, linestyle='--', label=key+'_loss')
		ax2.tick_params(axis='y', labelsize='small', length=6, width=3, which='both', direction='out')
		# ax1.spines['right'].set_visible(True)
		plt.axvline(x = ax2.get_xlim()[-1], color = 'black', linestyle = '--') 

		####
		plt.xlabel('Epochs')
		plt.ylabel('Cost (log scale)')
		plt.title('Acc. & Loss')
		plt.yscale('log')

		# {
		# 	'Acc.': Line2D([0,1],[0,1],linestyle='-', color='black'),
		# 	'Loss': Line2D([0,1],[0,1],linestyle='--', color='black'),
		# 	**{mpatches.Patch(color=val, label=key) for key, val in colour.items}
		# }
		plt.legend(handles=[
			Line2D([0,1],[0,1],linestyle='-', color='black', label='Acc.'),
			Line2D([0,1],[0,1],linestyle='--', color='black', label='Loss'),
			*(mpatches.Patch(color=val, label=key) for key, val in colour.items())
		])
		# plt.legend()
		plt.grid(True)
		fig.tight_layout()  # otherwise the right y-label is slightly clipped
		plt.savefig('./log/test_plot.pdf')


def plot_varying_datapoints(plot_data, fig_name, label, yscale):
	# Create a single plot for all training costs with a logarithmic scale
	with plt.xkcd():
		plt.figure(figsize=(8, 6))  # Set the figure size
		
		for n, _data in plot_data.items():
			epochs = range(1,len(_data)+1)
			test_cost = _data
			plt.plot(epochs, test_cost, marker='o', linestyle='-', label=f'{n} training points')

		plt.xlabel('Epochs')
		plt.ylabel(f'{label} ({yscale} scale)')
		plt.title(f'{label} over epochs for different training points (classification)')
		plt.yscale(yscale)
		plt.legend()
		plt.grid(True)
		plt.savefig(f'./log/{fig_name}.pdf')



