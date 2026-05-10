/**
 * Policy tab: load + introspect NN policies for closed-loop control.
 *
 * Placeholder for slice 6 (policy controller). Full plan:
 *  - List checkpoints under weights/ directory
 *  - Show input/output shapes once loaded
 *  - Optional observation preview (e.g. raw camera tensor going in)
 *  - Connection to a controller subprocess that runs the model
 */

export function PolicyTab() {
  return (
    <>
      <div className="row">
        <div className="label">Policy</div>
        <div className="help">
          Coming soon. Will host:
          <ul style={{ marginTop: 6, paddingLeft: 18 }}>
            <li>list of available NN checkpoints</li>
            <li>load / unload</li>
            <li>input + output shape introspection</li>
            <li>observation preview (camera tensor, joint state)</li>
            <li>action stream debugging</li>
          </ul>
          Lands in slice 6 of the controller refactor.
        </div>
      </div>
    </>
  );
}
