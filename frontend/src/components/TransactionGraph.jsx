import { useEffect, useRef, useState } from "react";
import { fetchTransactionGraph } from "../services/api";

const CANVAS_WIDTH = 700;
const CANVAS_HEIGHT = 450;

// Physics tuning constants - tweak these to change how "spread out" or
// "tight" the graph looks.
const REPULSION_STRENGTH = 2500; // how strongly nodes push each other apart
const SPRING_STRENGTH = 0.02; // how strongly connected nodes pull together
const SPRING_LENGTH = 90; // "ideal" edge length
const CENTERING_STRENGTH = 0.01; // pulls everything gently toward the center
const DAMPING = 0.85; // velocity decay per frame (prevents endless jitter)

export default function TransactionGraph() {
  const canvasRef = useRef(null);
  const nodesRef = useRef([]); // mutable simulation state - doesn't need re-render on every tick
  const edgesRef = useRef([]);
  const animationFrameRef = useRef(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [stats, setStats] = useState({ nodeCount: 0, edgeCount: 0 });

  // --- Step 1: fetch graph data from the backend once on mount ---
  useEffect(() => {
    async function loadGraph() {
      try {
        setLoading(true);
        const data = await fetchTransactionGraph(250);

        // Give each node a random starting position near the center -
        // the simulation will spread them out from here.
        const nodes = data.nodes.map((n) => ({
          ...n,
          x: CANVAS_WIDTH / 2 + (Math.random() - 0.5) * 100,
          y: CANVAS_HEIGHT / 2 + (Math.random() - 0.5) * 100,
          vx: 0,
          vy: 0,
        }));

        nodesRef.current = nodes;
        edgesRef.current = data.edges;
        setStats({ nodeCount: data.node_count, edgeCount: data.edge_count });
        setError(null);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    }

    loadGraph();
  }, []);

  // --- Step 2: run the physics simulation + drawing loop ---
  useEffect(() => {
    if (loading || error) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    function simulationStep() {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;

      // Repulsion: every node pushes every other node away.
      // O(n^2) - fine for a few hundred nodes, which is our scale here.
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i];
          const b = nodes[j];
          const dx = a.x - b.x;
          const dy = a.y - b.y;
          const distSq = Math.max(dx * dx + dy * dy, 1); // avoid divide-by-zero
          const force = REPULSION_STRENGTH / distSq;
          const dist = Math.sqrt(distSq);
          const fx = (dx / dist) * force;
          const fy = (dy / dist) * force;

          a.vx += fx;
          a.vy += fy;
          b.vx -= fx;
          b.vy -= fy;
        }
      }

      // Spring attraction: connected nodes pull toward each other,
      // proportional to how far they are from the "ideal" edge length.
      const nodeById = Object.fromEntries(nodes.map((n) => [n.id, n]));
      for (const edge of edges) {
        const source = nodeById[edge.source];
        const target = nodeById[edge.target];
        if (!source || !target) continue;

        const dx = target.x - source.x;
        const dy = target.y - source.y;
        const dist = Math.sqrt(dx * dx + dy * dy) || 1;
        const displacement = dist - SPRING_LENGTH;
        const force = displacement * SPRING_STRENGTH;
        const fx = (dx / dist) * force;
        const fy = (dy / dist) * force;

        source.vx += fx;
        source.vy += fy;
        target.vx -= fx;
        target.vy -= fy;
      }

      // Centering + velocity integration
      for (const node of nodes) {
        node.vx += (CANVAS_WIDTH / 2 - node.x) * CENTERING_STRENGTH;
        node.vy += (CANVAS_HEIGHT / 2 - node.y) * CENTERING_STRENGTH;

        node.vx *= DAMPING;
        node.vy *= DAMPING;

        node.x += node.vx;
        node.y += node.vy;

        // Keep nodes within the canvas bounds
        node.x = Math.max(20, Math.min(CANVAS_WIDTH - 20, node.x));
        node.y = Math.max(20, Math.min(CANVAS_HEIGHT - 20, node.y));
      }
    }

    function draw() {
      const nodes = nodesRef.current;
      const edges = edgesRef.current;
      const nodeById = Object.fromEntries(nodes.map((n) => [n.id, n]));

      ctx.clearRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);
      ctx.fillStyle = "#0b0f14"; // dark background
      ctx.fillRect(0, 0, CANVAS_WIDTH, CANVAS_HEIGHT);

      // Draw edges first, so nodes render on top of them
      for (const edge of edges) {
        const source = nodeById[edge.source];
        const target = nodeById[edge.target];
        if (!source || !target) continue;

        ctx.beginPath();
        ctx.moveTo(source.x, source.y);
        ctx.lineTo(target.x, target.y);

        if (edge.is_flagged) {
          // Glowing red for suspicious transaction chains
          ctx.strokeStyle = "rgba(255, 60, 60, 0.8)";
          ctx.shadowColor = "rgba(255, 40, 40, 0.9)";
          ctx.shadowBlur = 10;
          ctx.lineWidth = 1.5;
        } else {
          ctx.strokeStyle = "rgba(100, 180, 220, 0.25)";
          ctx.shadowBlur = 0;
          ctx.lineWidth = 1;
        }
        ctx.stroke();
      }
      ctx.shadowBlur = 0; // reset so it doesn't bleed into node drawing

      // Draw nodes on top
      for (const node of nodes) {
        const radius = node.is_flagged ? 7 : 5;

        ctx.beginPath();
        ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);

        if (node.is_flagged) {
          ctx.fillStyle = "#ff3c3c";
          ctx.shadowColor = "#ff2828";
          ctx.shadowBlur = 15;
        } else {
          ctx.fillStyle = "#4fd1ff";
          ctx.shadowBlur = 0;
        }
        ctx.fill();
        ctx.shadowBlur = 0;
      }
    }

    function tick() {
      simulationStep();
      draw();
      animationFrameRef.current = requestAnimationFrame(tick);
    }

    tick();

    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
    };
  }, [loading, error]);

  if (loading) {
    return <p style={{ color: "#aaa" }}>Loading transaction graph…</p>;
  }

  if (error) {
    return <p style={{ color: "#ff6b6b" }}>Failed to load graph: {error}</p>;
  }

  return (
    <div>
      <p style={{ color: "#8899a6", marginBottom: "8px" }}>
        {stats.nodeCount} accounts · {stats.edgeCount} transactions ·{" "}
        <span style={{ color: "#ff3c3c" }}>red = flagged</span>
      </p>
      <canvas
        ref={canvasRef}
        width={CANVAS_WIDTH}
        height={CANVAS_HEIGHT}
        style={{ border: "1px solid #1e2733", borderRadius: "8px" }}
      />
    </div>
  );
}