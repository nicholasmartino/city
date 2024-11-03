

class UrbanFabric:

	def __init__(self, parcels, streets):
		self.parcels = parcels
		self.streets = streets

	def is_valid(self):
		raise NotImplementedError()
