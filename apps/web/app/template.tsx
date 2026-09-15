/**
 * A template re-mounts on every navigation (a layout does not), so this is where the page
 * transition lives: each route fades and settles into place. The animation ends at the
 * page's normal state, so a page that never animates still looks right.
 */
export default function RouteTemplate({ children }: LayoutProps<"/">) {
  return <div className="page-enter">{children}</div>;
}
