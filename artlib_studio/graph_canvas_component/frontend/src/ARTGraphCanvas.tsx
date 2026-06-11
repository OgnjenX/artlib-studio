import { FrontendRendererArgs } from "@streamlit/component-v2-lib";
import {
  Background,
  Connection,
  Controls,
  Edge,
  Handle,
  MarkerType,
  MiniMap,
  Node,
  NodeProps,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useEdgesState,
  useNodesState,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  ChangeEvent,
  FC,
  ReactElement,
  useEffect,
  useRef,
  useState,
} from "react";

type PositionData = { x: number; y: number };

type ContextRule = {
  name: string;
  active: boolean;
  target_param: string;
  mode: string;
  value: number;
  duration: string;
  explanation?: string;
};

export type ModuleConfig = {
  id: string;
  type: "adapter" | "context";
  adapter?: string | null;
  params?: Record<string, number>;
  rules?: ContextRule[];
  position?: PositionData | null;
};

export type EdgeConfig = {
  source: string;
  target: string;
  type: string;
  transform?: {
    name: string;
    params?: Record<string, number>;
  } | null;
};

export type GraphConfig = {
  name: string;
  description: string;
  modules: ModuleConfig[];
  edges: EdgeConfig[];
  associations: unknown[];
  max_settling_steps: number;
};

type RuntimeModuleState = {
  selected_category?: number | null;
  category_count?: number;
  sample_count?: number;
  params?: Record<string, number>;
};

export type CanvasRuntime = {
  step_index: number;
  modules: Record<string, RuntimeModuleState>;
  active_modules: string[];
  active_edges: string[];
  last_event?: string | null;
};

export type ARTGraphCanvasStateShape = {
  graph: GraphConfig;
};

export type ARTGraphCanvasDataShape = {
  graph: GraphConfig;
  revision: number;
  runtime?: CanvasRuntime | null;
};

type SetStateValue = FrontendRendererArgs<
  ARTGraphCanvasStateShape,
  ARTGraphCanvasDataShape
>["setStateValue"];

type CanvasNodeData = {
  module: ModuleConfig;
  runtime?: RuntimeModuleState;
  active: boolean;
};

const edgeColor: Record<string, string> = {
  BOTTOM_UP: "#0f766e",
  TOP_DOWN_EXPECTATION: "#c2410c",
  MODULATORY: "#a16207",
  ASSOCIATIVE: "#0369a1",
  RESET_PROPAGATION: "#be123c",
};

const adapterDefinitions = {
  fuzzy_art: {
    name: "Fuzzy ART",
    defaults: { rho: 0.85, alpha: 0.0, beta: 1.0 },
  },
  gaussian_art: {
    name: "Gaussian ART",
    defaults: { rho: 0.5, alpha: 1e-10, sigma_init_scalar: 0.1 },
  },
  hypersphere_art: {
    name: "Hypersphere ART",
    defaults: { rho: 0.75, alpha: 0.0001, beta: 1.0, r_hat: 1.0 },
  },
} as const;

type AdapterKey = keyof typeof adapterDefinitions;

function adapterKey(module: ModuleConfig): AdapterKey {
  return module.adapter && module.adapter in adapterDefinitions
    ? (module.adapter as AdapterKey)
    : "fuzzy_art";
}

function moduleToNode(
  module: ModuleConfig,
  index: number,
  runtime?: CanvasRuntime | null,
): Node<CanvasNodeData> {
  const fallback = {
    x: 80 + (index % 3) * 300,
    y: 90 + Math.floor(index / 3) * 220,
  };
  return {
    id: module.id,
    type: module.type === "context" ? "contextNode" : "artNode",
    position: module.position ?? fallback,
    data: {
      module,
      runtime: runtime?.modules[module.id],
      active: runtime?.active_modules.includes(module.id) ?? false,
    },
  };
}

function edgeId(edge: EdgeConfig, index: number): string {
  return `${edge.source}:${edge.target}:${edge.type}:${index}`;
}

function configToEdges(
  edges: EdgeConfig[],
  runtime?: CanvasRuntime | null,
): Edge[] {
  return edges.map((edge, index) => {
    const id = edgeId(edge, index);
    const active = runtime?.active_edges.includes(
      `${edge.source}:${edge.target}`,
    );
    const color = edgeColor[edge.type] ?? "#475569";
    const isTopDown = edge.type === "TOP_DOWN_EXPECTATION";
    const isModulatory = edge.type === "MODULATORY";
    return {
      id,
      source: edge.source,
      target: edge.target,
      sourceHandle: isTopDown
        ? "top-down-out"
        : isModulatory
          ? "modulatory-out"
          : "bottom-up-out",
      targetHandle: isTopDown
        ? "expectation-in"
        : isModulatory
          ? "modulation-in"
          : "input-in",
      type: "smoothstep",
      label: edge.transform?.name
        ? `${edge.type} · ${edge.transform.name}`
        : edge.type,
      markerEnd: { type: MarkerType.ArrowClosed, color },
      animated: active,
      interactionWidth: 28,
      style: {
        stroke: color,
        strokeWidth: active ? 4 : 2,
      },
      labelStyle: { fill: color, fontWeight: 800, fontSize: 10 },
      labelBgStyle: {
        fill: "#f8faf7",
        fillOpacity: 0.94,
        stroke: color,
        strokeWidth: 0.5,
      },
      labelBgPadding: [6, 4] as [number, number],
      labelBgBorderRadius: 5,
      data: { configIndex: index },
    };
  });
}

const ARTNode: FC<NodeProps<Node<CanvasNodeData>>> = ({ data, selected }) => {
  const rho = data.runtime?.params?.rho ?? data.module.params?.rho;
  const definition = adapterDefinitions[adapterKey(data.module)];
  return (
    <div
      className={[
        "art-node",
        selected ? "selected" : "",
        data.active ? "active" : "",
      ].join(" ")}
    >
      <Handle
        id="input-in"
        className="port port-input"
        type="target"
        position={Position.Left}
        style={{ top: "32%" }}
        title="Bottom-up input"
      />
      <Handle
        id="modulation-in"
        className="port port-modulation"
        type="target"
        position={Position.Left}
        style={{ top: "52%" }}
        title="Modulatory input"
      />
      <Handle
        id="expectation-in"
        className="port port-expectation"
        type="target"
        position={Position.Left}
        style={{ top: "72%" }}
        title="Top-down expectation input"
      />
      <div className="node-kicker">{definition.name.toUpperCase()}</div>
      <strong>{data.module.id}</strong>
      <div className="node-grid">
        <span>rho</span><b>{rho ?? "n/a"}</b>
        <span>category</span>
        <b>{data.runtime?.selected_category ?? "none"}</b>
        <span>categories</span>
        <b>{data.runtime?.category_count ?? 0}</b>
        <span>samples</span><b>{data.runtime?.sample_count ?? 0}</b>
      </div>
      <span className="port-label port-label-bu">BU</span>
      <Handle
        id="bottom-up-out"
        className="port port-bottom-up"
        type="source"
        position={Position.Right}
        style={{ top: "32%" }}
        title="Bottom-up output"
      />
      <span className="port-label port-label-td">TD</span>
      <Handle
        id="top-down-out"
        className="port port-top-down"
        type="source"
        position={Position.Right}
        style={{ top: "72%" }}
        title="Top-down expectation output"
      />
    </div>
  );
};

const ContextNode: FC<NodeProps<Node<CanvasNodeData>>> = ({
  data,
  selected,
}) => {
  const rule = data.module.rules?.[0];
  return (
    <div
      className={[
        "context-node",
        selected ? "selected" : "",
        data.active ? "active" : "",
      ].join(" ")}
    >
      <div className="node-kicker">CONTEXT</div>
      <strong>{data.module.id}</strong>
      <div className="context-value">
        {rule?.target_param ?? "rho"} {rule?.mode ?? "set"}{" "}
        {rule?.value ?? "n/a"}
      </div>
      <small>{rule?.duration ?? "current_step"}</small>
      <Handle
        id="modulatory-out"
        className="port port-modulation"
        type="source"
        position={Position.Right}
        title="Modulatory output"
      />
    </div>
  );
};

const nodeTypes = { artNode: ARTNode, contextNode: ContextNode };

function defaultEdge(
  source: ModuleConfig,
  target: ModuleConfig,
  sourceHandle?: string | null,
): EdgeConfig {
  if (source.type === "context") {
    return { source: source.id, target: target.id, type: "MODULATORY" };
  }
  if (sourceHandle === "top-down-out") {
    return {
      source: source.id,
      target: target.id,
      type: "TOP_DOWN_EXPECTATION",
      transform: {
        name: "high_category_to_expectation",
        params: {},
      },
    };
  }
  return {
    source: source.id,
    target: target.id,
    type: "BOTTOM_UP",
    transform: {
      name: "selected_category_to_one_hot",
      params: { vector_size: 4 },
    },
  };
}

type Props = ARTGraphCanvasDataShape & { setStateValue: SetStateValue };

const ARTGraphCanvas: FC<Props> = ({
  graph,
  revision,
  runtime,
  setStateValue,
}): ReactElement => {
  const [config, setConfig] = useState<GraphConfig>(graph);
  const [nodes, setNodes, onNodesChange] = useNodesState(
    graph.modules.map((module, index) => moduleToNode(module, index, runtime)),
  );
  const [edges, setEdges, onEdgesChange] = useEdgesState(
    configToEdges(graph.edges, runtime),
  );
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState<string | null>(null);
  const [newAdapter, setNewAdapter] = useState<AdapterKey>("fuzzy_art");
  const [renameError, setRenameError] = useState<string | null>(null);
  const [graphNameDraft, setGraphNameDraft] = useState(graph.name);
  const seenRevision = useRef(revision);

  useEffect(() => {
    if (seenRevision.current !== revision) {
      seenRevision.current = revision;
      setConfig(graph);
      setNodes(
        graph.modules.map((module, index) =>
          moduleToNode(module, index, runtime),
        ),
      );
      setEdges(configToEdges(graph.edges, runtime));
      setSelectedNodeId(null);
      setSelectedEdgeId(null);
      setRenameError(null);
      setGraphNameDraft(graph.name);
      return;
    }
    setNodes((current) =>
      current.map((node) => ({
        ...node,
        data: {
          ...node.data,
          runtime: runtime?.modules[node.id],
          active: runtime?.active_modules.includes(node.id) ?? false,
        },
      })),
    );
    setEdges(configToEdges(config.edges, runtime));
  }, [graph, revision, runtime, setEdges, setNodes]);

  const publish = (next: GraphConfig) => {
    setConfig(next);
    setStateValue("graph", next);
  };

  const syncPositions = () => {
    const positions = new Map(nodes.map((node) => [node.id, node.position]));
    publish({
      ...config,
      modules: config.modules.map((module) => ({
        ...module,
        position: positions.get(module.id) ?? module.position,
      })),
    });
  };

  const onConnect = (connection: Connection) => {
    if (!connection.source || !connection.target) return;
    const source = config.modules.find(
      (module) => module.id === connection.source,
    );
    const target = config.modules.find(
      (module) => module.id === connection.target,
    );
    if (!source || !target || source.id === target.id) return;
    const edge = defaultEdge(source, target, connection.sourceHandle);
    const next = { ...config, edges: [...config.edges, edge] };
    setEdges(configToEdges(next.edges, runtime));
    publish(next);
  };

  const addModule = (type: "adapter" | "context") => {
    const base = type === "adapter" ? newAdapter : "context";
    let sequence = config.modules.length + 1;
    let id = `${base}_${sequence}`;
    while (config.modules.some((module) => module.id === id)) {
      sequence += 1;
      id = `${base}_${sequence}`;
    }
    const position = { x: 100 + sequence * 35, y: 120 + sequence * 25 };
    const module: ModuleConfig =
      type === "adapter"
        ? {
            id,
            type,
            adapter: newAdapter,
            params: { ...adapterDefinitions[newAdapter].defaults },
            rules: [],
            position,
          }
        : {
            id,
            type,
            adapter: null,
            params: {},
            rules: [
              {
                name: "vigilance_context",
                active: true,
                target_param: "rho",
                mode: "set",
                value: 0.95,
                duration: "current_step",
              },
            ],
            position,
          };
    const next = { ...config, modules: [...config.modules, module] };
    setNodes((current) => [
      ...current,
      moduleToNode(module, config.modules.length, runtime),
    ]);
    publish(next);
    setSelectedNodeId(id);
  };

  const removeSelection = () => {
    if (selectedNodeId) {
      const next = {
        ...config,
        modules: config.modules.filter(
          (module) => module.id !== selectedNodeId,
        ),
        edges: config.edges.filter(
          (edge) =>
            edge.source !== selectedNodeId && edge.target !== selectedNodeId,
        ),
      };
      setNodes((current) =>
        current.filter((node) => node.id !== selectedNodeId),
      );
      setEdges(configToEdges(next.edges, runtime));
      publish(next);
      setSelectedNodeId(null);
    } else if (selectedEdgeId) {
      const selected = edges.find((edge) => edge.id === selectedEdgeId);
      const index = selected?.data?.configIndex as number | undefined;
      if (index === undefined) return;
      const next = {
        ...config,
        edges: config.edges.filter((_, edgeIndex) => edgeIndex !== index),
      };
      setEdges(configToEdges(next.edges, runtime));
      publish(next);
      setSelectedEdgeId(null);
    }
  };

  const updateModule = (updated: ModuleConfig) => {
    const next = {
      ...config,
      modules: config.modules.map((module) =>
        module.id === updated.id ? updated : module,
      ),
    };
    setNodes((current) =>
      current.map((node) =>
        node.id === updated.id
          ? { ...node, data: { ...node.data, module: updated } }
          : node,
      ),
    );
    publish(next);
  };

  const renameGraph = (requestedName: string) => {
    const name = requestedName.trim();
    if (!name) {
      setRenameError("Network name cannot be empty.");
      return false;
    }
    setRenameError(null);
    if (name !== config.name) {
      setGraphNameDraft(name);
      publish({ ...config, name });
    }
    return true;
  };

  const renameModule = (oldId: string, requestedId: string) => {
    const id = requestedId.trim();
    if (!id) {
      setRenameError("Module name cannot be empty.");
      return false;
    }
    if (
      id !== oldId &&
      config.modules.some((module) => module.id === id)
    ) {
      setRenameError(`A module named "${id}" already exists.`);
      return false;
    }
    setRenameError(null);
    if (id === oldId) return true;

    const nextModules = config.modules.map((module) =>
      module.id === oldId ? { ...module, id } : module,
    );
    const nextEdges = config.edges.map((edge) => ({
      ...edge,
      source: edge.source === oldId ? id : edge.source,
      target: edge.target === oldId ? id : edge.target,
    }));
    const next = { ...config, modules: nextModules, edges: nextEdges };
    setNodes((current) =>
      current.map((node) =>
        node.id === oldId
          ? {
              ...node,
              id,
              data: {
                ...node.data,
                module: { ...node.data.module, id },
              },
            }
          : node,
      ),
    );
    setEdges(configToEdges(nextEdges, runtime));
    setSelectedNodeId(id);
    publish(next);
    return true;
  };

  const selectedModule = config.modules.find(
    (module) => module.id === selectedNodeId,
  );
  const selectedEdge = edges.find((edge) => edge.id === selectedEdgeId);
  const selectedEdgeIndex = selectedEdge?.data?.configIndex as
    | number
    | undefined;
  const selectedEdgeConfig =
    selectedEdgeIndex === undefined
      ? undefined
      : config.edges[selectedEdgeIndex];

  const updateSelectedEdge = (patch: Partial<EdgeConfig>) => {
    if (selectedEdgeIndex === undefined) return;
    const nextEdges = config.edges.map((edge, index) =>
      index === selectedEdgeIndex ? { ...edge, ...patch } : edge,
    );
    setEdges(configToEdges(nextEdges, runtime));
    publish({ ...config, edges: nextEdges });
  };

  return (
    <div className="studio-shell">
      <header className="canvas-toolbar">
        <div className="canvas-title-block">
          <span className="eyebrow">ART COMPOSITION CANVAS</span>
          <input
            className="canvas-title-input"
            aria-label="Network name"
            type="text"
            value={graphNameDraft}
            onChange={(event) => {
              setGraphNameDraft(event.target.value);
              if (renameError) {
                setRenameError(null);
              }
            }}
            onBlur={(event) => {
              if (!renameGraph(event.target.value)) {
                setGraphNameDraft(config.name);
              }
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter") {
                event.currentTarget.blur();
              }
            }}
          />
        </div>
        <div className="toolbar-actions">
          <select
            className="adapter-picker"
            aria-label="ART model to add"
            value={newAdapter}
            onChange={(event) =>
              setNewAdapter(event.target.value as AdapterKey)
            }
          >
            {Object.entries(adapterDefinitions).map(([key, definition]) => (
              <option key={key} value={key}>
                {definition.name}
              </option>
            ))}
          </select>
          <button onClick={() => addModule("adapter")}>+ Add ART</button>
          <button onClick={() => addModule("context")}>+ Context</button>
          <button
            className="danger"
            disabled={!selectedNodeId && !selectedEdgeId}
            onClick={removeSelection}
          >
            Delete selected
          </button>
        </div>
        <div className="runtime-chip">
          step {runtime?.step_index ?? 0}
          <span>{runtime?.last_event ?? "idle"}</span>
        </div>
      </header>

      <div className="canvas-body">
        <main className="flow-panel">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={onConnect}
            onNodeClick={(_, node) => {
              setSelectedNodeId(node.id);
              setSelectedEdgeId(null);
            }}
            onEdgeClick={(_, edge) => {
              setSelectedEdgeId(edge.id);
              setSelectedNodeId(null);
            }}
            onPaneClick={() => {
              setSelectedNodeId(null);
              setSelectedEdgeId(null);
            }}
            onNodeDragStop={syncPositions}
            fitView
            minZoom={0.25}
            maxZoom={2}
            deleteKeyCode={null}
          >
            <Background color="#cbd5e1" gap={24} size={1.5} />
            <MiniMap
              nodeColor={(node) =>
                node.type === "contextNode" ? "#eab308" : "#14b8a6"
              }
              maskColor="rgba(241,245,249,.72)"
            />
            <Controls />
          </ReactFlow>
          <div className="canvas-hint">
            Drag nodes. Use BU for bottom-up and TD for top-down expectation.
            Select an edge to edit it.
          </div>
        </main>

        <aside className="inspector">
          <div className="inspector-heading">
            <span>INSPECTOR</span>
            <strong>
              {selectedModule?.id ??
                (selectedEdgeConfig
                  ? `${selectedEdgeConfig.source} → ${selectedEdgeConfig.target}`
                  : "Select a node or edge")}
            </strong>
          </div>

          {selectedModule && (
            <div className="field-stack module-identity">
              <label>
                Module name
                <input
                  key={`module-name-${selectedModule.id}`}
                  type="text"
                  defaultValue={selectedModule.id}
                  onBlur={(event) => {
                    if (
                      !renameModule(
                        selectedModule.id,
                        event.target.value,
                      )
                    ) {
                      event.target.value = selectedModule.id;
                    }
                  }}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.currentTarget.blur();
                    }
                  }}
                />
              </label>
            </div>
          )}

          {renameError && (
            <div className="field-error" role="alert">
              {renameError}
            </div>
          )}

          {selectedModule?.type === "adapter" && (
            <div className="field-stack">
              <label>
                ART model
                <select
                  value={adapterKey(selectedModule)}
                  onChange={(event) => {
                    const adapter = event.target.value as AdapterKey;
                    updateModule({
                      ...selectedModule,
                      adapter,
                      params: { ...adapterDefinitions[adapter].defaults },
                    });
                  }}
                >
                  {Object.entries(adapterDefinitions).map(
                    ([key, definition]) => (
                      <option key={key} value={key}>
                        {definition.name}
                      </option>
                    ),
                  )}
                </select>
              </label>
              <label>
                Vigilance rho
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.01"
                  value={selectedModule.params?.rho ?? 0.85}
                  onChange={(event) =>
                    updateModule({
                      ...selectedModule,
                      params: {
                        ...selectedModule.params,
                        rho: Number(event.target.value),
                      },
                    })
                  }
                />
                <output>{selectedModule.params?.rho ?? 0.85}</output>
              </label>
              <NumberField
                label="Choice alpha"
                value={
                  selectedModule.params?.alpha ??
                  adapterDefinitions[adapterKey(selectedModule)].defaults.alpha
                }
                step={
                  adapterKey(selectedModule) === "gaussian_art"
                    ? 1e-10
                    : 0.001
                }
                onChange={(value) =>
                  updateModule({
                    ...selectedModule,
                    params: { ...selectedModule.params, alpha: value },
                  })
                }
              />
              {adapterKey(selectedModule) !== "gaussian_art" && (
                <NumberField
                  label="Learning beta"
                  value={selectedModule.params?.beta ?? 1}
                  step={0.01}
                  onChange={(value) =>
                    updateModule({
                      ...selectedModule,
                      params: { ...selectedModule.params, beta: value },
                    })
                  }
                />
              )}
              {adapterKey(selectedModule) === "gaussian_art" && (
                <NumberField
                  label="Initial sigma"
                  value={selectedModule.params?.sigma_init_scalar ?? 0.1}
                  step={0.01}
                  onChange={(value) =>
                    updateModule({
                      ...selectedModule,
                      params: {
                        ...selectedModule.params,
                        sigma_init_scalar: value,
                      },
                    })
                  }
                />
              )}
              {adapterKey(selectedModule) === "hypersphere_art" && (
                <NumberField
                  label="Maximum radius"
                  value={selectedModule.params?.r_hat ?? 1}
                  step={0.1}
                  onChange={(value) =>
                    updateModule({
                      ...selectedModule,
                      params: { ...selectedModule.params, r_hat: value },
                    })
                  }
                />
              )}
            </div>
          )}

          {selectedModule?.type === "context" && (
            <div className="field-stack">
              <NumberField
                label="Context value"
                value={selectedModule.rules?.[0]?.value ?? 0.95}
                step={0.01}
                onChange={(value) => {
                  const rule = selectedModule.rules?.[0];
                  if (!rule) return;
                  updateModule({
                    ...selectedModule,
                    rules: [{ ...rule, value }],
                  });
                }}
              />
              <label>
                Duration
                <select
                  value={
                    selectedModule.rules?.[0]?.duration ?? "current_step"
                  }
                  onChange={(event) => {
                    const rule = selectedModule.rules?.[0];
                    if (!rule) return;
                    updateModule({
                      ...selectedModule,
                      rules: [{ ...rule, duration: event.target.value }],
                    });
                  }}
                >
                  <option value="current_step">current_step</option>
                  <option value="persistent">persistent</option>
                </select>
              </label>
            </div>
          )}

          {selectedEdgeConfig && (
            <div className="field-stack">
              <label>
                Edge type
                <select
                  value={selectedEdgeConfig.type}
                  onChange={(event) =>
                    updateSelectedEdge({ type: event.target.value })
                  }
                >
                  {[
                    "BOTTOM_UP",
                    "TOP_DOWN_EXPECTATION",
                    "MODULATORY",
                    "ASSOCIATIVE",
                    "RESET_PROPAGATION",
                  ].map((type) => (
                    <option key={type}>{type}</option>
                  ))}
                </select>
              </label>
              <label>
                Transform
                <select
                  value={selectedEdgeConfig.transform?.name ?? "none"}
                  onChange={(event) => {
                    const name = event.target.value;
                    updateSelectedEdge({
                      transform:
                        name === "none"
                          ? null
                          : {
                              name,
                              params:
                                name === "selected_category_to_one_hot"
                                  ? { vector_size: 4 }
                                  : {},
                            },
                    });
                  }}
                >
                  <option value="none">none</option>
                  <option value="selected_category_to_one_hot">
                    selected_category_to_one_hot
                  </option>
                  <option value="selected_category_to_scalar_vector">
                    selected_category_to_scalar_vector
                  </option>
                  <option value="selected_category_to_activation_vector">
                    selected_category_to_activation_vector
                  </option>
                  <option value="high_category_to_expectation">
                    high_category_to_expectation
                  </option>
                </select>
              </label>
            </div>
          )}

          {!selectedModule && !selectedEdgeConfig && (
            <div className="empty-inspector">
              <div className="pulse-ring" />
              <p>
                Select a module to tune ART parameters, or select an edge to
                change its signal semantics.
              </p>
            </div>
          )}
        </aside>
      </div>
    </div>
  );
};

const NumberField: FC<{
  label: string;
  value: number;
  step: number;
  onChange: (value: number) => void;
}> = ({ label, value, step, onChange }) => (
  <label>
    {label}
    <input
      type="number"
      value={value}
      step={step}
      onChange={(event: ChangeEvent<HTMLInputElement>) =>
        onChange(Number(event.target.value))
      }
    />
  </label>
);

const WrappedCanvas: FC<Props> = (props) => (
  <ReactFlowProvider>
    <ARTGraphCanvas {...props} />
  </ReactFlowProvider>
);

export default WrappedCanvas;
