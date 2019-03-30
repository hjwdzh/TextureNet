class Params:
	def __init__(self, num_point=8192, num_classes=21, batch_size=4,
		learning_rate=1e-3, momentum=0.9, decay_step=200000, decay_rate=0.7,
		bn_init_decay=0.5, bn_decay_rate=0.5, bn_decay_step=2000000, bn_decay_clip=0.99):
		self.num_point = num_point
		self.num_classes = num_classes
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		self.momentum = momentum
		self.decay_step = decay_step
		self.decay_rate = decay_rate
		self.bn_init_decay = bn_init_decay
		self.bn_decay_rate = bn_decay_rate
		self.bn_decay_step = decay_step
		self.bn_decay_clip = bn_decay_clip