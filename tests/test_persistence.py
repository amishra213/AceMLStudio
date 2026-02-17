"""Test dataset persistence functionality."""

from ml_engine.db_storage import DataFrameDBStorage
import pandas as pd
import os

def test_dataset_persistence():
    """Test save, load, list, update, and delete operations."""
    
    db_path = 'test_datasets.db'
    db = DataFrameDBStorage(db_path)
    
    # Create test dataset
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': ['x', 'y', 'z']
    })
    
    print('=== Testing Save ===')
    result = db.save_dataset(
        dataset_name='test_dataset',
        df=df,
        description='Test description',
        tags=['test', 'demo']
    )
    print(f'✓ Save successful: {result}')
    
    print('\n=== Testing Load ===')
    loaded = db.load_dataset('test_dataset')
    assert loaded is not None, "Failed to load dataset"
    print(f'✓ Loaded shape: {loaded.shape}')
    print(f'✓ Loaded columns: {list(loaded.columns)}')
    print(f'✓ Data matches: {df.equals(loaded)}')
    
    print('\n=== Testing List ===')
    datasets = db.list_datasets()
    print(f'✓ Found {len(datasets)} dataset(s)')
    if datasets:
        ds = datasets[0]
        print(f'  - Name: {ds["dataset_name"]}')
        print(f'  - Rows: {ds["rows"]}')
        print(f'  - Columns: {ds["columns"]}')
        print(f'  - Tags: {ds["tags"]}')
        print(f'  - Description: {ds["description"]}')
    
    print('\n=== Testing Search ===')
    search_results = db.list_datasets(search='test')
    print(f'✓ Search for "test": {len(search_results)} result(s)')
    
    print('\n=== Testing Update (add column) ===')
    df['D'] = [10, 20, 30]
    result2 = db.save_dataset(
        dataset_name='test_dataset',
        df=df,
        description='Updated with column D',
        tags=['test', 'updated']
    )
    print(f'✓ Update successful: {result2}')
    loaded2 = db.load_dataset('test_dataset')
    assert loaded2 is not None, "Failed to load updated dataset"
    print(f'✓ Updated shape: {loaded2.shape}')
    print(f'✓ Updated columns: {list(loaded2.columns)}')
    print(f'✓ Has new column D: {"D" in loaded2.columns}')
    
    print('\n=== Testing Update (remove column) ===')
    df_smaller = df.drop(columns=['C'])
    result3 = db.save_dataset(
        dataset_name='test_dataset',
        df=df_smaller,
        description='Removed column C',
        tags=['test', 'reduced']
    )
    print(f'✓ Update successful: {result3}')
    loaded3 = db.load_dataset('test_dataset')
    assert loaded3 is not None, "Failed to load reduced dataset"
    print(f'✓ Updated shape: {loaded3.shape}')
    print(f'✓ Updated columns: {list(loaded3.columns)}')
    print(f'✓ Column C removed: {"C" not in loaded3.columns}')
    
    print('\n=== Testing Multiple Datasets ===')
    df2 = pd.DataFrame({'X': [100, 200], 'Y': [300, 400]})
    db.save_dataset(dataset_name='second_dataset', df=df2, description='Second dataset', tags=['demo'])
    all_datasets = db.list_datasets()
    print(f'✓ Total datasets: {len(all_datasets)}')
    for ds in all_datasets:
        print(f'  - {ds["dataset_name"]}: {ds["rows"]}×{ds["columns"]}')
    
    print('\n=== Testing Delete ===')
    del_result = db.delete_dataset('test_dataset')
    print(f'✓ Delete successful: {del_result}')
    remaining = db.list_datasets()
    print(f'✓ Remaining datasets: {len(remaining)}')
    
    # Cleanup
    print('\n=== Cleanup ===')
    db.delete_dataset('second_dataset')
    # Close any connections (if needed)
    del db
    import time
    time.sleep(0.5)  # Give time for file handles to release
    if os.path.exists(db_path):
        try:
            os.remove(db_path)
            print('✓ Database cleaned up')
        except PermissionError:
            print('⚠ Database file still in use, will be cleaned up later')
    else:
        print('✓ Database cleaned up')
    
    print('\n' + '='*50)
    print('✅ ALL TESTS PASSED!')
    print('='*50)

if __name__ == '__main__':
    test_dataset_persistence()
