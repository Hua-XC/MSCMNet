from solver.lr_scheduler import LRSchedulerWithWarmup


def adjust_learning_rate(optimizer, lr_scheduler_model):

    return LRSchedulerWithWarmup(
        optimizer,
        # step
        milestones=(25,90,120),
        #milestones=(20,60,80),

        gamma=0.1,
        warmup_factor=0.1,

        warmup_epochs=10,      
        warmup_method="linear",
        total_epochs=150,
        mode=lr_scheduler_model,
        target_lr=0,
        power=0.9,
    )
