import os, shutil

def save_saved_model(model, out_directory):
    print('Saving saved model...')

    saved_model_path = os.path.join(out_directory, 'saved_model')
    model.save(saved_model_path, save_format='tf')
    shutil.make_archive(saved_model_path,
                        'zip',
                        root_dir=os.path.dirname(saved_model_path),
                        base_dir='saved_model')

    print('Saving saved model OK')
    print('')
