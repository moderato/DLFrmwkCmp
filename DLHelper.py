import h5py

def init_h5py(filename, epoch_num, max_total_batch):
    f = h5py.File(filename, 'w')
        
    try:
        # config group for some common params
        config = f.create_group('config')
        config.attrs["total_epochs"] = epoch_num

        cost = f.create_group('cost')
        loss = cost.create_dataset('loss', (epoch_num,))
        loss.attrs['time_markers'] = 'epoch_freq'
        loss.attrs['epoch_freq'] = 1
        train = cost.create_dataset('train', (max_total_batch,)) # Set size to maximum theoretical value
        train.attrs['time_markers'] = 'minibatch'

        t = f.create_group('time')
        loss = t.create_dataset('loss', (epoch_num,))
        train = t.create_group('train')
        start_time = train.create_dataset("start_time", (1,))
        start_time.attrs['units'] = 'seconds'
        end_time = train.create_dataset("end_time", (1,))
        end_time.attrs['units'] = 'seconds'
        train_batch = t.create_dataset('train_batch', (max_total_batch,)) # Same as above

        # Mark which batches are the end of an epoch
        time_markers = f.create_group('time_markers')
        time_markers.attrs['epochs_complete'] = epoch_num
        train_batch = time_markers.create_dataset('minibatch', (epoch_num,))
    except Exception as e:
        self.f.close() # Avoid hdf5 runtime error or os error
        raise e # Catch the exception to close the file, then raise it to stop the program

    return f