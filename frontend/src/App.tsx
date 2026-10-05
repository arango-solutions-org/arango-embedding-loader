import { BookOpen, Database, ListChecks, Upload } from "lucide-react";
import { useEffect, useState, type ReactNode } from "react";

import { ConnectionPanel, ConnectionStatus } from "./components/ConnectionPanel";
import { Collections } from "./pages/Collections";
import { Jobs } from "./pages/Jobs";
import { LoadWizard } from "./pages/LoadWizard";

type Route = "load" | "collections" | "jobs";

function useHashRoute(): [Route, (r: Route) => void] {
  const read = (): Route => {
    const r = window.location.hash.replace("#/", "") as Route;
    return ["load", "collections", "jobs"].includes(r) ? r : "load";
  };
  const [route, setRoute] = useState<Route>(read);
  useEffect(() => {
    const onHash = () => setRoute(read());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  return [route, (r) => (window.location.hash = `#/${r}`)];
}

function NavItem({
  active,
  label,
  icon,
  onClick,
  href,
}: {
  active?: boolean;
  label: string;
  icon: ReactNode;
  onClick?: () => void;
  href?: string;
}) {
  const cls = `flex w-full flex-col items-center gap-1 rounded-md px-1 py-2.5 text-[11px] text-white ${
    active ? "bg-white/20" : "hover:bg-white/15"
  }`;
  return href ? (
    <a className={cls} href={href} target="_blank" rel="noreferrer">
      {icon}
      {label}
    </a>
  ) : (
    <button type="button" className={cls} onClick={onClick} aria-current={active ? "page" : undefined}>
      {icon}
      {label}
    </button>
  );
}

export default function App() {
  const [route, setRoute] = useHashRoute();
  const [showConnection, setShowConnection] = useState(false);

  return (
    <div className="flex h-full">
      <nav className="flex w-20 shrink-0 flex-col items-center gap-2 bg-black px-2 py-4" aria-label="Main">
        <img src="/favicon.svg" alt="" className="mb-4 size-9" />
        <NavItem active={route === "load"} label="Load" icon={<Upload className="size-5" />} onClick={() => setRoute("load")} />
        <NavItem
          active={route === "collections"}
          label="Collections"
          icon={<Database className="size-5" />}
          onClick={() => setRoute("collections")}
        />
        <NavItem active={route === "jobs"} label="Jobs" icon={<ListChecks className="size-5" />} onClick={() => setRoute("jobs")} />
        <div className="mt-auto w-full">
          <NavItem label="API docs" icon={<BookOpen className="size-5" />} href="/docs" />
        </div>
      </nav>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between gap-4 border-b border-line px-6 py-3">
          <span className="text-sm font-semibold">
            Arango <span className="text-arango-brand">Embedding Loader</span>
          </span>
          <ConnectionStatus onOpen={() => setShowConnection(true)} />
        </header>
        <main className="flex-1 overflow-y-auto px-6 py-6">
          <div className="mx-auto max-w-6xl">
            {route === "load" && <LoadWizard />}
            {route === "collections" && <Collections />}
            {route === "jobs" && <Jobs />}
          </div>
        </main>
      </div>
      {showConnection && <ConnectionPanel onClose={() => setShowConnection(false)} />}
    </div>
  );
}
