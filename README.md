# Research-Placement
This repository contains the current code for the active learning framework to accelarate Finite Element simulation for the multiscale analysis of neo-hookian metamaterials.

Purpose:
This framework aims to use machine learning to accelarate the optimisation steps during the energy minimisation process used in non-linear response analysis, this research is currently ongoing, however, after testing we have decided to move to a spectral approach instead of this FEA based approach.

Features:
Currently the framework uses geometric distance to trigger simulator query, note: Mahalanobis filtering is the more effective approach and so if this approach is ever returned to this is a change to be made. A GNN using attention layers is used to predict strain energy from boundary node displacements on multiscale elements.