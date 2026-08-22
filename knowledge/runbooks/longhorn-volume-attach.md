---
title: Longhorn volume stuck in "attaching"
type: runbook
tags: [longhorn, storage, csi, node-reboot]
---

# Runbook: Longhorn volume stuck in "attaching"

## Symptom
A pod is stuck in `ContainerCreating`; events show `FailedAttachVolume` or
`Multi-Attach error for volume`. The Longhorn UI shows the volume in state
`attaching` or `faulted`.

## First checks (read-only)
1. Confirm the workload's namespace and the PVC/PV name from the pod events.
2. Check `longhorn-system` before blaming the app: list `longhorn-manager` pods
   and the volume's status. A degraded/faulted volume is a Longhorn problem, not
   an app problem.
3. Check node health — a recently rebooted or `NotReady` node commonly leaves a
   volume attached to the dead node.

## Common root causes
- **Stale attachment after node reboot.** The volume is still recorded as
  attached to a node that went away. Longhorn usually reconciles within a few
  minutes; if it does not, the `volumeattachment` is stuck.
- **Replica scheduling failure.** Not enough healthy nodes/disks to satisfy the
  replica count. The volume stays `degraded` and may refuse to attach.
- **Instance-manager pod crashloop** on the target node.

## Escalation
If the volume is `faulted`, do not delete anything — capture the Longhorn
support bundle and escalate to the storage owner. Deletion risks data loss.
