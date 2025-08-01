import { app } from "/scripts/app.js";

const _ID = "ConditionalSelectionNode";
const _PREFIX = "input";
const _TYPE = "*";

const _SWITCH_ID = "GeneralSwitch";

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

// Extension for GeneralSwitch (Switch Any) node
app.registerExtension({
	name: 'Yaser-nodes.' + _SWITCH_ID,
	async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== _SWITCH_ID) {
            return;
        }

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
            const stackTrace = new Error().stack;
            
            // Handle loading and pasting scenarios
            if(stackTrace.includes('loadGraphData') || stackTrace.includes('pasteFromClipboard')) {
                if(this.widgets?.[0]) {
                    this.widgets[0].options.max = this.inputs.length - 2; // Subtract select and sel_mode
                    this.widgets[0].value = Math.min(this.widgets[0].value, this.widgets[0].options.max);
                    if(this.widgets[0].options.max > 0 && this.widgets[0].value == 0)
                        this.widgets[0].value = 1;
                }
                return;
            }

            if(!link_info)
                return;

            if(type == 2) {
                // Output connection
                if(connected && index == 0){
                    // Prevent connecting to Reroute nodes (similar to Impact Pack behavior)
                    if(app.graph._nodes_by_id[link_info.target_id]?.type == 'Reroute') {
                        app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
                    }

                    if(this.outputs[0].type == '*'){
                        if(link_info.type == '*' && app.graph.getNodeById(link_info.target_id).slots[link_info.target_slot].type != '*') {
                            app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
                        } else {
                            // Propagate type from output to inputs
                            this.outputs[0].type = link_info.type;
                            this.outputs[0].label = link_info.type;
                            this.outputs[0].name = link_info.type;

                            for(let i in this.inputs) {
                                let input_i = this.inputs[i];
                                if(input_i.name != 'select' && input_i.name != 'sel_mode')
                                    input_i.type = link_info.type;
                            }
                        }
                    }
                }
                return;
            } else {
                // Input connection
                // Prevent connecting from Reroute nodes (similar to Impact Pack behavior)
                if(app.graph._nodes_by_id[link_info.origin_id].type == 'Reroute')
                    this.disconnectInput(link_info.target_slot);

                // Skip control inputs
                if(this.inputs[index].name == 'select' || this.inputs[index].name == 'sel_mode')
                    return;

                // Handle type propagation for dynamic inputs
                if(this.inputs[0].type == '*' || this.outputs[0].type == '*'){
                    const node = app.graph.getNodeById(link_info.origin_id);
                    let origin_type = node.outputs[link_info.origin_slot]?.type;

                    if(origin_type == '*' && app.graph.getNodeById(link_info.origin_id).slots[link_info.origin_slot].type != '*') {
                        this.disconnectInput(link_info.target_slot);
                        return;
                    }

                    // Set type for all dynamic inputs and output
                    for(let i in this.inputs) {
                        let input_i = this.inputs[i];
                        if(input_i.name != 'select' && input_i.name != 'sel_mode')
                            input_i.type = origin_type;
                    }

                    this.outputs[0].type = origin_type;
                    this.outputs[0].label = origin_type;
                    this.outputs[0].name = origin_type;
                }
            }

            // Handle dynamic input management
            if (!connected && (this.inputs.length > 3)) { // More than select, sel_mode, and one input
                if(!stackTrace.includes('LGraphNode.prototype.connect') && 
                   !stackTrace.includes('LGraphNode.connect') && 
                   !stackTrace.includes('loadGraphData') &&
                   this.inputs[index].name != 'select' && 
                   this.inputs[index].name != 'sel_mode') {
                    this.removeInput(index);
                }
            }

            // Rename inputs to maintain sequential numbering
            let slot_i = 1;
            for (let i = 0; i < this.inputs.length; i++) {
                let input_i = this.inputs[i];
                if(input_i.name != 'select' && input_i.name != 'sel_mode') {
                    input_i.name = `input${slot_i}`;
                    slot_i++;
                }
            }

            // Add new input slot when connecting
            if(connected && this.inputs[index].name != 'select' && this.inputs[index].name != 'sel_mode') {
                this.addInput(`input${slot_i}`, this.outputs[0].type);
            }

            // Update widget max value
            if(this.widgets?.[0]) {
                this.widgets[0].options.max = this.inputs.length - 2; // Subtract select and sel_mode
                this.widgets[0].value = Math.min(this.widgets[0].value, this.widgets[0].options.max);
                if(this.widgets[0].options.max > 0 && this.widgets[0].value == 0)
                    this.widgets[0].value = 1;
            }
        }

        return nodeType;
    },
})
