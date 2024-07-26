
# W1D2_Tutorial1.ipynb

import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

def plot_single_model_train_metrics(value_dict:dict, _show:bool=False, _save:bool=True, _fig_name:str='./log/test_plot.pdf'):
	# Create a single plot for all training costs with a logarithmic scale
	if 'train_loss' in value_dict.keys():
		_cost_legend = 'loss'
	elif 'train_E' in value_dict.keys():
		_cost_legend = 'E'
	else:
		raise ValueError(f"Train cost metric not recognized in {value_dict.keys()} was expecting something in ('train_loss', 'train_E').")
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
		ax2.set_ylabel('Cost (log scale)')  # we already handled the x-label with ax1
		for key, val in colour.items():
			ax2.plot(t, value_dict[key+f'_{_cost_legend}'], color=val, linestyle='--', label=key+f'_{_cost_legend}')
		ax2.tick_params(axis='y', labelsize='small', length=6, width=3, which='both', direction='out')
		# ax1.spines['right'].set_visible(True)
		plt.axvline(x = ax2.get_xlim()[-1], color = 'black', linestyle = '--') 

		####
		plt.xlabel('Epochs')
		# plt.ylabel('Cost (log scale)')
		plt.title('Acc. & Cost')
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
		if _save:
			plt.savefig(_fig_name)
		if _show:
			plt.show()

def plot_varying_datapoints(plot_data:dict, fig_name:str, label:str, yscale:str, _show:bool=False, _save:bool=True):
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

		if _save:
			plt.savefig(fig_name)
		if _show:
			plt.show()

def plot_varying_datapoints_all_models(plot_data:dict, fig_name:str, label:str, yscale:str, _show:bool=False, _save:bool=True):
	with plt.xkcd():
		plt.figure(figsize=(8, 6))  # Set the figure 
		_cmap = plt.cm.get_cmap('viridis', len(plot_data))
		_linestyles = ('solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'dotted', 'densely dotted', 'long dash with offset', 'loosely dashed', 'densely dashed', 'loosely dashdotted', 'dashdotted', 'densely dashdotted', 'dashdotdotted', 'loosely dashdotdotted', 'densely dashdotdotted')
		
		for _colour_idx, (_model, _data_dict) in enumerate(plot_data.items()):
			for _linestyle_idx, (n, _data) in enumerate(_data_dict.items()):
				epochs = range(1,len(_data)+1)
				test_cost = _data
				# plt.plot(epochs, test_cost, color=_cmap(_colour_idx), marker='o', linestyle=_linestyles[_linestyle_idx], label=f'{n} tr. pts ({_model})')
				plt.plot(epochs, test_cost, color=_cmap(_colour_idx), linestyle=_linestyles[_linestyle_idx], label=f'{n} tr. pts ({_model})')

		plt.xlabel('Epochs')
		plt.ylabel(f'{label} ({yscale} scale)')
		plt.title(f'{label} over epochs for different training points (classification)')
		plt.yscale(yscale)
		# plt.legend()
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True)
		plt.tight_layout()

		if _save:
			plt.savefig(fig_name)
		if _show:
			plt.show()

def plot_varying_datapoints_all_model_types(plot_data:dict, fig_name:str, label:str, yscale:str, _show:bool=False, _save:bool=True):
	with plt.xkcd():
		plt.figure(figsize=(8, 6))  # Set the figure 
		_cmap = plt.cm.get_cmap('viridis', len(plot_data))
		_linestyles = ('solid', 'dotted', 'dashed', 'dashdot', 'loosely dotted', 'dotted', 'densely dotted', 'long dash with offset', 'loosely dashed', 'densely dashed', 'loosely dashdotted', 'dashdotted', 'densely dashdotted', 'dashdotdotted', 'loosely dashdotdotted', 'densely dashdotdotted')
		
		for _colour_idx, (_model, _data_dict) in enumerate(plot_data.items()):
			for _linestyle_idx, (n, _data) in enumerate(_data_dict.items()):
				epochs = range(1,len(_data)+1)
				test_cost = _data
				# plt.plot(epochs, test_cost, color=_cmap(_colour_idx), marker='o', linestyle=_linestyles[_linestyle_idx], label=f'{n} tr. pts ({_model})')
				plt.plot(epochs, test_cost, color=_cmap(_colour_idx), linestyle=_linestyles[_linestyle_idx], label=f'{n} tr. pts ({_model})')

		plt.xlabel('Epochs')
		plt.ylabel(f'{label} ({yscale} scale)')
		plt.title(f'{label} over epochs for different training points (classification)')
		plt.yscale(yscale)
		# plt.legend()
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.grid(True)
		plt.tight_layout()

		if _save:
			plt.savefig(fig_name)
		if _show:
			plt.show()



# ax.fill_between(x, y1, y2, alpha=.5, linewidth=0)


