import { app } from "/scripts/app.js";

const _ID = "ConditionalSelectionNode";
const _PREFIX = "input";
const _TYPE = "*";

app.registerExtension({
	name: 'Yaser-nodes.' + _ID,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _ID) {
            return
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = async function () {
            const me = onNodeCreated?.apply(this);
            this.addInput(_PREFIX, _TYPE);
            const slot = this.inputs[this.inputs.length - 1];
            if (slot) {
                slot.color_off = "#666";
            }
            return me;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (slotType, slot_idx, event, link_info, node_slot) {
            const me = onConnectionsChange?.apply(this, arguments);

            if (slotType === 1) { // Input
                if (link_info && event) { // Connect
                    const fromNode = this.graph._nodes.find(
                        (otherNode) => otherNode.id == link_info.origin_id
                    )

                    if (fromNode) {
                        const parent_link = fromNode.outputs[link_info.origin_slot];
                        if (parent_link) {
                            node_slot.type = parent_link.type;
                            node_slot.name = `${_PREFIX}_`;
                        }
                    }
                } else { // Disconnect
                    this.removeInput(slot_idx);
                }

                let idx = 0;
                let slot_tracker = {};
                for(const slot of this.inputs) {
                    if (slot.link === null) {
                        this.removeInput(idx);
                        continue;
                    }
                    idx += 1;
                    const name = slot.name.split('_')[0];
                    let count = (slot_tracker[name] || 0) + 1;
                    slot_tracker[name] = count;
                    slot.name = `${name}_${count}`;
                }

                let last = this.inputs[this.inputs.length - 1];
                if (last === undefined || (last.name != _PREFIX || last.type != _TYPE)) {
                    this.addInput(_PREFIX, _TYPE);
                    last = this.inputs[this.inputs.length - 1];
                    if (last) {
                        last.color_off = "#666";
                    }
                }
                this?.graph?.setDirtyCanvas(true);
                return me;
            }
        }
        return nodeType;
    },
})
