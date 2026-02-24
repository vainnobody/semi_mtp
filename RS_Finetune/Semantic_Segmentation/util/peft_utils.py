from peft.tuners.semift import RSMT


def set_task_trainable(model, task_id):
    for name, module in model.named_modules():
        if isinstance(module, RSMT):
            module.set_task_id(task_id)


def set_trainable(model):
    for name, module in model.named_modules():
        if isinstance(module, RSMT):
            module.set_trainable()


def set_task_id(model, task_id):
    for name, module in model.named_modules():
        if isinstance(module, RSMT):
            module.set_task_id(task_id)


def set_task(model, task_id):
    set_task_trainable(model, task_id)
    set_task_id(model, task_id)
    return model
