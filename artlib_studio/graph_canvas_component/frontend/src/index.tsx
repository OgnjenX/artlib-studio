import {
  FrontendRenderer,
  FrontendRendererArgs,
} from "@streamlit/component-v2-lib";
import { createRoot, Root } from "react-dom/client";

import ARTGraphCanvas, {
  ARTGraphCanvasDataShape,
  ARTGraphCanvasStateShape,
} from "./ARTGraphCanvas";
import "./index.css";

const reactRoots: WeakMap<FrontendRendererArgs["parentElement"], Root> =
  new WeakMap();

const ARTGraphCanvasRoot: FrontendRenderer<
  ARTGraphCanvasStateShape,
  ARTGraphCanvasDataShape
> = (args) => {
  const { data, parentElement, setStateValue } = args;
  const rootElement = parentElement.querySelector(".react-root");
  if (!rootElement) {
    throw new Error("ART graph canvas root element not found");
  }

  let reactRoot = reactRoots.get(parentElement);
  if (!reactRoot) {
    reactRoot = createRoot(rootElement);
    reactRoots.set(parentElement, reactRoot);
  }

  reactRoot.render(
    <ARTGraphCanvas
      graph={data.graph}
      revision={data.revision}
      runtime={data.runtime}
      setStateValue={setStateValue}
    />,
  );

  return () => {
    const root = reactRoots.get(parentElement);
    if (root) {
      root.unmount();
      reactRoots.delete(parentElement);
    }
  };
};

export default ARTGraphCanvasRoot;
