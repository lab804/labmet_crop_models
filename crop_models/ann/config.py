from crop_models.ann.ann_sets import *
from neurolab.train import train_gd, train_gdm, train_gda, train_gdx, train_rprop, train_bfgs, train_cg, train_ncg
from neurolab.error import MSE, SSE, SAE, MAE, CEE

mlp_train_algorithm = FrozenDict({"train_gd": train_gd,
                                  "train_gdm": train_gdm,
                                  "train_gda": train_gda,
                                  "train_gdx": train_gdx,
                                  "train_rprop": train_rprop,
                                  "train_bfgs": train_bfgs,
                                  "train_cg": train_cg,
                                  "train_ncg": train_ncg
                                  })

error_functions = FrozenDict({"mse": MSE(),
                              "sse": SSE(),
                              "sae": SAE(),
                              "mae": MAE(),
                              "cee": CEE()}
                             )

