import sys
from City.Zoning import Zone, Envelope, Box, Trapeze


if 'win32' in sys.platform:
	directory = f'C:/Users/nichmar.stu/Google Drive/Python/urban-zoning/data'
else:
	directory = '/Users/nicholasmartino/Google Drive/Python/urban-zoning/data'

zone_objects = {

	'C-1': Zone(
		directory=directory,
		name='C-1',
		land_use='commercial',
		frontage='N/A',
		envelopes={'standard': Envelope([Box(4.6, 0.6, 0.6, 0)])},
		fsr={'standard': 1.2, 'use:dwelling': 0.75},
		max_coverage=0.6
	),

	'C-2': Zone(
		directory=directory,
		name='C-2',
		land_use='commercial',
		envelopes={
			'standard': Envelope([Box(4.6, 0.6, 0.6, 0), Box(10.7, 0.6, 0.6, 6.1), Box(13.8, 3, 3, 10.7)]),
			'adjacent:side:r': Envelope([Box(10.7, 0.6, 0.6, 3.7), Box(13.8, 3, 3, 10.7)])
		},
		fsr={'standard': 1.2, 'use:dwelling': 0.75},
		max_coverage=0.6
	),

	'C-2B': Zone(
		directory=directory,
		name='C-2B',
		land_use='commercial',
		frontage=15.3,
		envelopes={
			'standard': Envelope([Box(12.2, 0, 0, 3.1)]),
			'adjacent:side:r': Envelope([Box(12.2, side=1.5)]),
			'adjacent:side:rm': Envelope([Box(12.2, side=0.9)]),
			'adjacent:front:ew': Envelope([Trapeze(anchor='north_line', angle=30, base_height=7.3)]),
			'accessory': Envelope([Box(3.7, rear=3.1, rear_anchor='centerline')]),
		},
		fsr={'standard': 1, 'flexible': 3, 'use:accessory': 0.15},
	),

	'C-2C': Zone(
		directory=directory,
		name='C-2C',
		land_use='commercial',
		frontage=15.3,
		envelopes={
			'standard': Envelope([Box(10.7, 0.6, 0.9, 3.1)]),
			'adjacent:side:r': Envelope([Box(12.2, side=1.5)]),
			'adjacent:side:rm': Envelope([Box(12.2, side=0.9)]),
			'adjacent:front:ew': Envelope([Trapeze(anchor='north_line', angle=30, base_height=7.3)]),
		},
		fsr={'standard': 3, 'use:office': 1.2, 'use:residential': 1.5},
	),

	'C-3A': Zone(
		directory=directory,
		name='C-3A',
		land_use='commercial',
		envelopes={
			'standard': Envelope([Box(9.2, 0, 0.9, 3.1, rear_anchor='centerline')]),
			'adjacent:r': Envelope([Box(side=1.5)]),
			'accessory': Envelope([Box(3.7, pitch_height=4.6, rear=3.1)])
		},
		fsr={'standard': 1, 'flexible': 3, 'use:accessory': 0.15},
	),

	'CD-1': Zone(
		directory=directory,
		name='CD-1',
		land_use='mixed',
		envelopes={
			'standard': Envelope([Box(9.2, 0.5, 0.9, 3.1, rear_anchor='centerline'), Box(36, 1.5, 1.5, 5)]),
			'adjacent:r': Envelope([Box(side=1.5)]),
			'accessory': Envelope([Box(3.7, pitch_height=4.6, rear=3.1)])
		},
		fsr={'standard': 1, 'flexible': 3, 'use:accessory': 0.15},
	),

	'CD-1 (761)': Zone(
		directory=directory,
		name='CD-1 (761)',
		land_use='mixed',
		envelopes={
			'standard': Envelope([Box(9.2, 0, 0.9, 3.1, rear_anchor='centerline')]),
			'adjacent:r': Envelope([Box(side=1.5)]),
			'accessory': Envelope([Box(3.7, pitch_height=4.6, rear=3.1)])
		},
		fsr={'standard': 1, 'flexible': 3, 'use:accessory': 0.15},
		max_coverage=0.6
	),

	'I-2': Zone(
		directory=directory,
		name='I-2',
		land_use='industrial',
		envelopes={
			'standard': Envelope([Box(18.3, 0.6, 7.6, 3.1), Box(30.5, 3.7, 10.7, 3.1)]),
			'accessory': Envelope([Box(3.7, pitch_height=4.6)])
		},
		fsr={'standard': 1, 'use:manufacturing': 3},
		max_coverage=0.6
	),

	'I-3': Zone(
		directory=directory,
		name='I-3',
		land_use='industrial',
		envelopes={
			'standard': Envelope([Box(18.3, 0.6, 7.6, 3.1), Box(30.5, 3.7, 10.7, 3.1)]),
			'accessory': Envelope([Box(3.7, pitch_height=4.6)])
		},
		fsr={'standard': 1, 'use:manufacturing': 3},
		max_coverage=0.6
	),

	'RT-3': Zone(
		directory=directory,
		name='RT-3',
		land_use='residential',
		envelopes={
			'standard': Envelope([Box(10.7, 3.7, 1.5, 20)]),
		},
		fsr={'standard': 0.6},
		max_coverage=0.6
	)
}
