import tensorboard

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator('.')
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

# E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
w_times, step_nums, vals = zip(*event_acc.Scalars('Accuracy'))
