import tensorboard as tb

exp_id = "c1KCv3X3QvGwaXfgX1c4tg"
exp = tb.data.experimental.ExperimentFromDev(exp_id)
df = exp.get_scalars()
df


