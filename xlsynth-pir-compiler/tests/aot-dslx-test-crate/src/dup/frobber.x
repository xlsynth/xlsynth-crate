// SPDX-License-Identifier: Apache-2.0

import foo.widget as foo_widget;
import bar.widget as bar_widget;

pub fn frob_widget(widget: foo_widget::Widget) -> bar_widget::Widget {
    bar_widget::Widget { widget_id: widget.widget_id + u8:1 }
}
