from .nodes import WorkflowCanaryCapture, WorkflowCanaryCheck

NODE_CLASS_MAPPINGS = {
    "WorkflowCanaryCapture": WorkflowCanaryCapture,
    "WorkflowCanaryCheck": WorkflowCanaryCheck,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "WorkflowCanaryCapture": "WorkflowCanary · Capture Baseline / 保存基线",
    "WorkflowCanaryCheck": "WorkflowCanary · Check Baseline / 更新后验收",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
