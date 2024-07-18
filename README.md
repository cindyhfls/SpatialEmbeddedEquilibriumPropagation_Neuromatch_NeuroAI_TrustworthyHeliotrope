# SpatialEmbeddedEquilibriumPropagation_Neuromatch_NeuroAI_TrustworthyHeliotrope
Code for Trustworthy Heliotrope group "Equilibrium" team



```python
# Run MLP with BP
python main_RR.py --c_energy cross_entropy --seed 2019 --epochs 2
# Run 'cond_gaussian' energy model
python main_RR.py --energy cond_gaussian --c_energy cross_entropy --seed 2019 --epochs 2
```