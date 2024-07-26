import numpy as np
import matplotlib.pyplot as plt



def plot_maps(model_features:dict, model_name:str, show:bool=False, save:bool=True, fig_name:str=''):
	"""(From W1D3_T1)
	Plots representational dissimilarity matrices (RDMs) across different layers of a model.

	Inputs:
	- model_features (dict): a dictionary where keys are layer names and values are numpy arrays representing RDMs for each layer.
	- model_name (str): the name of the model being visualized.
	"""
	with plt.xkcd():

		fig = plt.figure(figsize=(14, 4))
		fig.suptitle(f"RDMs across layers for {model_name}")
		# and we add one plot per reference point
		gs = fig.add_gridspec(1, len(model_features))
		fig.subplots_adjust(wspace=0.2, hspace=0.2)

		for l in range(len(model_features)):

			layer = list(model_features.keys())[l]
			map_ = np.squeeze(model_features[layer])

			if len(map_.shape) < 2:
				map_ = map_.reshape( (int(np.sqrt(map_.shape[0])), int(np.sqrt(map_.shape[0]))) )

			map_ = map_ / np.max(map_)

			ax = plt.subplot(gs[0,l])
			ax_ = ax.imshow(map_, cmap='magma_r')
			ax.set_title(f'{layer}')
			ax.set_xlabel("input index")
			if l==0:
				ax.set_ylabel("input index")

		fig.subplots_adjust(right=0.9)
		cbar_ax = fig.add_axes([1.01, 0.18, 0.01, 0.53])
		cbar = fig.colorbar(ax_, cax=cbar_ax)
		cbar.set_label('Dissimilarity', rotation=270, labelpad=15)

		if save:
			plt.savefig(fig_name)
		if show:
			plt.show()


def plot_label_rdm(rdms_dict, show:bool=False, save:bool=True, fig_name:str=''):
	# plot rdm
	with plt.xkcd():
		plt.figure(figsize=(6, 4))
		plt.imshow(rdms_dict['labels']/rdms_dict['labels'].max(), cmap='magma_r')
		plt.title("RDM of the labels")
		plt.xlabel("input index")
		plt.ylabel("input index")
		cbar = plt.colorbar()
		cbar.set_label('Dissimilarity', rotation=270, labelpad=15)
		
		if save:
			plt.savefig(fig_name)
		if show:
			plt.show()

