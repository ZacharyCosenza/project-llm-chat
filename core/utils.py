import os

def is_ddp():
    return int(os.environ.get('RANK', -1)) != -1

def print0(s="",**kwargs):
    if is_ddp():
        ddp_rank = int(os.environ.get('RANK', 0))
        if ddp_rank == 0:
            print(s, **kwargs)
    else:
        print(s, **kwargs)

def get_dist_info():
    if is_ddp():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1
    